import csv
import json
import os

from Bio.PDB import PDBParser, MMCIFIO, MMCIFParser
from Bio.SeqUtils import seq1

# ================= 1. Global constants =================

PREFIX_HEADER = """data_new_structure
#
_pdbx_audit_revision_history.ordinal             1
_pdbx_audit_revision_history.data_content_type   'Structure model'
_pdbx_audit_revision_history.major_revision      1
_pdbx_audit_revision_history.minor_revision      0
_pdbx_audit_revision_history.revision_date       2026-04-09
#
"""

# ================= 2. Core functions =================

def convert_pdb_to_cif_force_chain_a(pdb_path, cif_path):
    """ Convert PDB to CIF and force the ID of the first chain to 'A' """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("new_structure", pdb_path)
    for model in structure:
        for chain in model:
            chain.id = 'A'
            break # Only the first chain is processed
        break
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(cif_path)
    with open(cif_path, 'r') as f:
        original_content = f.read()
    new_content = PREFIX_HEADER + "\n" + original_content
    with open(cif_path, 'w') as f:
        f.write(new_content)

def parse_cif_for_seq_and_mapping(cif_path):
    """ Parse CIF to obtain sequence and mapping """
    parser = MMCIFParser(QUIET=True)
    structure_id = os.path.basename(cif_path).split('.')[0]
    structure = parser.get_structure(structure_id, cif_path)
    model = next(iter(structure))
    chain = next(iter(model))
    sequence = ""
    auth_to_index = {}
    current_index = 0
    for residue in chain:
        if residue.id[0] == ' ':
            res_name = residue.resname
            aa_code = seq1(res_name)
            sequence += aa_code
            auth_id = residue.id[1]
            auth_to_index[auth_id] = current_index
            current_index += 1
    return sequence, auth_to_index

def get_insertions(orig_seq: str, full_seq: str):
    """ Find insertion segments of full_seq relative to orig_seq using the two-pointer technique """
    insertions = []
    o_idx, f_idx = 0, 0
    while o_idx < len(orig_seq) and f_idx < len(full_seq):
        if orig_seq[o_idx] == full_seq[f_idx]:
            o_idx += 1
            f_idx += 1
        else:
            f_start = f_idx
            # Find the next matching point
            while f_idx < len(full_seq) and (o_idx >= len(orig_seq) or full_seq[f_idx] != orig_seq[o_idx]):
                f_idx += 1
            ins_seq = full_seq[f_start:f_idx]
            insertions.append((o_idx, ins_seq))
    if f_idx < len(full_seq):
        insertions.append((len(orig_seq), full_seq[f_idx:]))
    return insertions

def update_index(old_idx: int, insertions):
    """ Update the index value based on insertion segment information """
    shift = 0
    for ins_pos, ins_seq in insertions:
        if old_idx >= ins_pos:
            shift += len(ins_seq)
    return old_idx + shift

def generate_torchfold_json(name, cif_path, full_sequence, cif_sequence, json_output_path):
    """ Generate TorchFold JSON, handle queryIndices offset caused by internal gaps """
    remote_cif_path = cif_path
    
    # Get insertion fragment information
    insertions = get_insertions(cif_sequence, full_sequence)
    
    # The original CIF indices are [0, 1, 2, ..., len(cif_sequence)-1]
    # They need to be mapped one-to-one to the indices in full_sequence
    query_indices = [update_index(i, insertions) for i in range(len(cif_sequence))]
    template_indices = list(range(len(cif_sequence)))
    
    msa_string = f">query\n{full_sequence}\n"
    
    json_data = {
        "dialect": "TorchFold",
        "version": 1,
        "name": name,
        "sequences":[
            {
                "protein": {
                    "id": "A",
                    "sequence": full_sequence,
                    "modifications": [],
                    "unpairedMsa": msa_string,
                    "pairedMsa": msa_string,
                    "templates":[
                        {
                            "mmcifPath": remote_cif_path,
                            "queryIndices": query_indices,
                            "templateIndices": template_indices
                        }
                    ]
                }
            },
            {
                "protein": {
                    "id": "B",
                    "sequence": "",
                    "modifications":[],
                    "unpairedMsa": "",
                    "pairedMsa": "",
                    "templates":[]
                }
            }
        ],
        "modelSeeds": [10],
        "bondedAtomPairs": None,
        "userCCD": None
    }
    with open(json_output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    return insertions

def process_hotspots(raw_hotspots_str, auth_to_index, insertions=None):
    """ Process Hotspot string mapping, considering offsets caused by internal gaps """
    raw_list = [h.strip() for h in raw_hotspots_str.split(',') if h.strip()]
    valid_raw, indices_0, indices_1 = [], [], []
    for h_str in raw_list:
        try:
            auth_id = int(h_str)
            if auth_id in auth_to_index:
                # Get the original 0-based index in the CIF sequence
                orig_idx_0 = auth_to_index[auth_id]
                # Update the index if insertion fragment information exists
                idx_0 = update_index(orig_idx_0, insertions) if insertions else orig_idx_0
                idx_1 = idx_0 + 1
                valid_raw.append(str(auth_id))
                indices_0.append(str(idx_0))
                indices_1.append(str(idx_1))
        except ValueError:
            continue
    return ",".join(valid_raw), ",".join(indices_0), ",".join(indices_1)

# ================= 3. the main preprocess =================

def preprosess_pdb(input_pdb_file, input_hotspot_string, output_root_dir, target_full_seq_csvfile):
    """
    Preprocess the PDB file and generate the corresponding JSON file and Hotspot index string.
    
    :param input_pdb_file: Absolute path to the input PDB file
    :param input_hotspot_string: Hotspot site string, e.g., "9,54,55"
    :param output_root_dir: Root directory for output results
    """

    output_cif_dir = os.path.join(output_root_dir, "target_cif")
    output_json_dir = os.path.join(output_root_dir, "target_json")

    # 1. Load gap information (handle special quotes that may exist in the CSV)
    gap_dict = {}
    if os.path.exists(target_full_seq_csvfile):
        with open(target_full_seq_csvfile, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                target = row['target'].strip(' "“”')
                seq = row['real_sequence'].strip()
                gap_dict[target] = seq

    # 2. Create output directory
    os.makedirs(output_cif_dir, exist_ok=True)
    os.makedirs(output_json_dir, exist_ok=True)

    results_for_csv = []
    print(f"=== Start processing: {input_pdb_file} ===")

    # 3. Read Hotspot CSV and process single file
    raw_hotspots = input_hotspot_string
    pdb_path = input_pdb_file
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"[ERROR] Can not find PDB file: {pdb_path}")
    target_name = os.path.basename(pdb_path).split('.')[0]
    print(f"target_name = {target_name}")

    try:
        # Step A: Convert CIF and modify chain ID
        cif_filename = f"{target_name}.cif"
        cif_path = os.path.join(output_cif_dir, cif_filename)
        convert_pdb_to_cif_force_chain_a(pdb_path, cif_path)

        # Step B: Parse CIF to get sequence and mapping
        cif_sequence, auth_to_index = parse_cif_for_seq_and_mapping(cif_path)

        # Step C: Determine the final sequence
        insertions = None
        if target_name in gap_dict:
            full_sequence = gap_dict[target_name]
            print(f"  [Info] Using preset full sequence (length: {len(full_sequence)})")
        else:
            full_sequence = cif_sequence

        # Step D: Generate JSON and get insertion fragment information
        json_path = os.path.join(output_json_dir, f"{target_name}.json")
        insertions = generate_torchfold_json(target_name, cif_path, full_sequence, cif_sequence, json_path)

        # Step E: Calculate the mapped Hotspot indices
        v_raw, i0, _ = process_hotspots(raw_hotspots, auth_to_index, insertions)

        print("\n=== Processing completed ===")
        print(f"CIF directory: {output_cif_dir}")
        print(f"JSON directory: {output_json_dir}")
        
        return json_path, i0
    except Exception as e:
        print(f"  [Failed] Processing {target_name} failed: {e}")
