import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

import pandas as pd
from Bio import PDB
from Bio.PDB import MMCIFIO
from Bio.PDB import Structure, Model
from tqdm import tqdm


def split_by_total_length(df, num_jobs):
    """
    Split into num_jobs batches with balanced total_length sums.
    """
    df = df.sort_values("total_length", ascending=False).reset_index(drop=True)
    total_sum = df["total_length"].sum()
    target_sum = total_sum / num_jobs

    groups = []
    current_group = []
    current_sum = 0

    for _, row in df.iterrows():
        val = int(row["total_length"])
        if current_sum + val > target_sum and len(groups) < num_jobs - 2:
            groups.append(pd.DataFrame(current_group))
            current_group = [row]
            current_sum = val
        else:
            current_group.append(row)
            current_sum += val

    len_last_group = len(current_group)
    index = int(len_last_group / 2.2)
    groups.append(pd.DataFrame(current_group[:index]))
    groups.append(pd.DataFrame(current_group[index:]))

    return groups


def format_msa_sequence(sequence):
    return f">query\n{sequence}\n"


def get_chain_sequences_from_row(row):
    chain_sequences = []
    chain_columns = [
        col for col in row.index
        if col.startswith("chain_") and col.endswith("_seq")
    ]
    for col in chain_columns:
        if pd.notna(row[col]) and row[col] != "":
            chain_id = col.split("_")[1]
            chain_sequences.append((chain_id, row[col]))
    return chain_sequences


protein_letters_3to1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    "MSE": "M",
}


def get_sequence_from_chain(chain):
    sequence = ""
    for residue in chain:
        if residue.id[0] == " ":
            resname = residue.get_resname().upper()
            sequence += protein_letters_3to1.get(resname, "X")
    return sequence


def process_single_pdb(args):
    input_pdb, output_dir_cif = args
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb)
        base_name = os.path.splitext(os.path.basename(input_pdb))[0]

        chain_sequences = {}
        merged_sequence = ""

        for chain in structure[0]:
            chain_id = chain.id
            sequence = get_sequence_from_chain(chain)
            chain_sequences[chain_id] = sequence
            merged_sequence += sequence

            new_structure = Structure.Structure("new_structure")
            new_model = Model.Model(0)
            new_structure.add(new_model)
            new_model.add(chain.copy())

            cif_io = MMCIFIO()
            cif_io.set_structure(new_structure)
            cif_output = os.path.join(output_dir_cif, f"{base_name}_chain_{chain_id}.cif")
            cif_io.save(cif_output)

        return base_name, chain_sequences, len(merged_sequence)

    except Exception as e:
        print(f"Error processing {input_pdb}: {str(e)}")
        return None, None, None


def generate_json_files(tasks):
    row, cif_dir, output_dir = tasks
    complex_name = row["complex"]
    chain_sequences = get_chain_sequences_from_row(row)

    if not chain_sequences:
        print(f"Warning: No valid chain sequences for {complex_name}")
        return None

    sequences = []
    for chain_id, sequence in chain_sequences:
        cif_filename = f"{complex_name}_chain_{chain_id}.cif"
        cif_path = os.path.join(cif_dir, cif_filename)
        if not os.path.exists(cif_path):
            print(f"Warning: {cif_filename} not found")
            continue
        sequences.append({
            "protein": {
                "id": chain_id,
                "sequence": sequence,
                "modifications": [],
                "unpairedMsa": format_msa_sequence(sequence),
                "pairedMsa": format_msa_sequence(sequence),
                "templates": [{
                    "mmcifPath": cif_path,
                    "queryIndices": list(range(len(sequence))),
                    "templateIndices": list(range(len(sequence))),
                }],
            }
        })

    if not sequences:
        print(f"Warning: No valid sequence data for {complex_name}")
        return None

    json_data = {
        "dialect": "TorchFold",
        "version": 1,
        "name": complex_name,
        "sequences": sequences,
        "modelSeeds": [10],
        "bondedAtomPairs": None,
        "userCCD": None,
    }

    output_filename = f"{complex_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return output_filename


def get_seq_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir_cif", type=str, required=True)
    parser.add_argument("--save_csv", type=str, required=True)
    parser.add_argument("--output_dir_json", type=str, required=True)
    parser.add_argument("--batch_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_jobs", type=int, default=1)
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers else mp.cpu_count() - 4
    os.makedirs(args.output_dir_cif, exist_ok=True)
    os.makedirs(args.output_dir_json, exist_ok=True)

    # Phase 1: Parallel PDB Processing
    pdb_files = list(Path(args.input_dir).glob("*.pdb"))
    pdb_args = [(str(f), args.output_dir_cif) for f in pdb_files]

    sequences_dict = {}
    with mp.Pool(processes=min(num_workers, 10)) as pool:
        results = list(tqdm(pool.imap(process_single_pdb, pdb_args), total=len(pdb_files), desc="Parsing PDBs"))

    for base_name, chain_sequences, length in results:
        if base_name:
            sequences_dict[base_name] = {"sequences": chain_sequences, "length": length}

    # Aggregate all unique chain IDs
    all_chain_ids = set()
    for entry in sequences_dict.values():
        all_chain_ids.update(entry["sequences"].keys())
    all_chain_ids = sorted(list(all_chain_ids), key=str)

    # Build DataFrame
    rows = []
    for complex_name, entry in sequences_dict.items():
        row = {"complex": complex_name, "total_length": entry["length"]}
        for cid in all_chain_ids:
            row[f"chain_{cid}_seq"] = entry["sequences"].get(cid, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["complex", "total_length"] + [c for c in df.columns if c not in ["complex", "total_length"]]
    df = df[cols]
    df.to_csv(args.save_csv, index=False)

    # Phase 2: Parallel JSON Generation
    json_tasks = [(row, args.output_dir_cif, args.output_dir_json) for _, row in df.iterrows()]
    with mp.Pool(processes=min(num_workers, 10)) as pool:
        list(tqdm(pool.imap(generate_json_files, json_tasks), total=len(json_tasks), desc="Generating JSONs"))

    # Phase 3: Batch Partitioning and Symlinking
    batch_json = f"{args.batch_dir}/json"
    batch_pdb = f"{args.batch_dir}/pdb"
    df = df.sample(frac=1).reset_index(drop=True)

    subs = split_by_total_length(df, args.num_jobs)
    for i, sub in enumerate(subs):
        if sub.empty:
            continue
        mx = sub["total_length"].max()
        name = f"batch_{i}_{mx}"
        bd_json, bd_pdb = os.path.join(batch_json, name), os.path.join(batch_pdb, name)
        os.makedirs(bd_json, exist_ok=True)
        os.makedirs(bd_pdb, exist_ok=True)

        for _, r in sub.iterrows():
            cid = r["complex"]
            for ext, src_dir, dest_dir in [
                (".pdb", args.input_dir, bd_pdb),
                (".json", args.output_dir_json, bd_json),
            ]:
                src = os.path.join(src_dir, f"{cid}{ext}")
                if os.path.exists(src):
                    dest_path = os.path.join(dest_dir, f"{cid}{ext}")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.symlink(src, dest_path)
        print(f"Batch {name}: {len(sub)} complexes")


if __name__ == "__main__":
    get_seq_main()
