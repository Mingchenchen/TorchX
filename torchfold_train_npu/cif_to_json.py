import argparse
import json
from pathlib import Path

from tqdm import tqdm

from torchfold.constants import mmcif_names
from torchfold.structure import parsing as struc_parsing

# Standard mapping from three-letter amino acid codes to single letters
AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'SEC': 'U', 'PYL': 'O',
}

# Nucleic-acid residue mapping
NUCLEIC_3TO1 = {
    'A': 'A', 'G': 'G', 'C': 'C', 'U': 'U', 'T': 'T',
    'DA': 'A', 'DG': 'G', 'DC': 'C', 'DT': 'T',
    'AMP': 'A', 'GMP': 'G', 'CMP': 'C', 'UMP': 'U', 'TMP': 'T',
}

# Combined residue lookup table
RESIDUE_MAP = {**AA_3TO1, **NUCLEIC_3TO1}

def convert_single_cif(cif_path: str, json_dir: str):
    """
    Parse a single CIF file, extract sequences and SMILES, and dump JSON.

    Args:
        cif_path: Path to the input CIF file.
        json_dir: Directory path for the output JSON files.
    """
    cif_file = Path(cif_path)
    print(f"\nProcessing file: {cif_file.name}...")

    # 1. Read and parse the CIF file
    try:
        with open(cif_file, 'r') as f:
            cif_string = f.read()
        struc = struc_parsing.from_mmcif(cif_string)
    except Exception as e:
        print(f"  Error parsing CIF file: {e}")
        return

    # Prepare the final structured output bundle
    output_data = {
        "name": cif_file.stem,
        "sequences": [],
        "modelSeeds": [1],
        "dialect": "torchfold",
        "version": 1
    }
    
    chain_sequences = struc.chain_res_name_sequence(include_missing_residues=False)
    chain_id_to_type = dict(zip(struc.chains_table.id, struc.chains_table.type))

    # 2. Traverse all chains and extract sequence/ligand info
    for chain_id in struc.chains:
        chain_type = chain_id_to_type.get(chain_id)
        
        if chain_type == mmcif_names.NON_POLYMER_CHAIN:
            res_names = chain_sequences.get(chain_id, [])
            if res_names:
                ligand_name = res_names[0]
                chem_comp_entry = struc.chemical_components_data.chem_comp.get(ligand_name)
                if chem_comp_entry and hasattr(chem_comp_entry, 'pdbx_smiles') and chem_comp_entry.pdbx_smiles:
                    smiles = chem_comp_entry.pdbx_smiles
                    print(f"  Found Ligand '{ligand_name}' in chain '{chain_id}': Extracting SMILES...")
                    chain_entry = {"ligand": {"id": chain_id, "smiles": smiles}}
                    output_data["sequences"].append(chain_entry)
                else:
                    print(f"  Warning: Ligand '{ligand_name}' in chain '{chain_id}' has no SMILES string.")
        else:
            res_names = chain_sequences.get(chain_id, [])
            if not res_names:
                continue

            sequence = ''
            unknown_residues = set()
            for res_name in res_names:
                res_name_upper = res_name.upper()
                one_letter_code = RESIDUE_MAP.get(res_name_upper)
                if one_letter_code:
                    sequence += one_letter_code
                else:
                    sequence += 'X'
                    unknown_residues.add(res_name)
            
            if unknown_residues:
                print(f"    Warning: Unknown residue types in chain '{chain_id}': {', '.join(unknown_residues)}.")

            if not sequence:
                continue

            chain_info = {"id": chain_id, "sequence": sequence}
            chain_entry = {}
            if chain_type == mmcif_names.PROTEIN_CHAIN:
                print(f"  Found Protein chain '{chain_id}': Converting to sequence...")
                chain_entry = {"protein": chain_info}
            elif chain_type == mmcif_names.RNA_CHAIN:
                print(f"  Found RNA chain '{chain_id}': Converting to sequence...")
                chain_entry = {"rna": chain_info}
            elif chain_type == mmcif_names.DNA_CHAIN:
                print(f"  Found DNA chain '{chain_id}': Converting to sequence...")
                chain_entry = {"dna": chain_info}
            
            if chain_entry:
                output_data["sequences"].append(chain_entry)

    # 3. Build the output path and write the JSON payload
    output_dir = Path(json_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_filename = cif_file.stem + '.json'
    output_path = output_dir / json_filename

    print(f"  Writing output to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a directory of CIF structure files to JSON files containing sequences and SMILES strings, compatible with TorchFold input format."
    )
    parser.add_argument(
        '--cif_dir', 
        type=str, 
        required=True, 
        help="Path to the directory containing input CIF files (e.g., /path/to/cif_folder)"
    )
    parser.add_argument(
        '--json_dir', 
        type=str, 
        required=True, 
        help="Directory to save the output JSON files (e.g., /path/to/json_output)"
    )

    args = parser.parse_args()

    cif_dir = Path(args.cif_dir)
    if not cif_dir.is_dir():
        print(f"Error: Input CIF directory not found at {args.cif_dir}")
        return

    cif_files = list(cif_dir.glob('*.cif')) + list(cif_dir.glob('*.mmcif'))
    if not cif_files:
        print(f"No .cif or .mmcif files found in {cif_dir}")
        return

    print(f"Found {len(cif_files)} CIF files to process.")

    for cif_file in tqdm(cif_files, desc="Overall Progress"):
        convert_single_cif(str(cif_file), args.json_dir)

    print("\nAll files processed successfully!")


if __name__ == '__main__':
    main()
