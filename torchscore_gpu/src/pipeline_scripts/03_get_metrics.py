import json
import multiprocessing as mp
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm

from ipsae_calculator import load_tf_pae_and_chains, calculate_ipsae


def get_chains_from_pdb(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_path)
    model = structure[0]
    chains = [chain.id for chain in model.get_chains()]
    return sorted(set(chains))


def get_interface_res_from_pdb(pdb_file, chain1='A', chain2='B', dist_cutoff=10):
    chain_coords = defaultdict(dict)

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                chain_id = line[21].strip()
                residue_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                if atom_name == "CA":
                    chain_coords[chain_id][residue_id] = np.array([x, y, z])

    chain_1_res = sorted(chain_coords[chain1].keys())
    chain_2_res = sorted(chain_coords[chain2].keys())

    chain_1_coords = np.array([chain_coords[chain1][res] for res in chain_1_res])
    chain_2_coords = np.array([chain_coords[chain2][res] for res in chain_2_res])

    dist = np.sqrt(np.sum((chain_1_coords[:, None, :] - chain_2_coords[None, :, :])**2, axis=2))
    interface_residues = np.where(dist < dist_cutoff)

    interface_1 = sorted(set(chain_1_res[i] for i in interface_residues[0]))
    interface_2 = sorted(set(chain_2_res[i] for i in interface_residues[1]))
    return interface_1, interface_2


def extract_token_chain_and_res_ids(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("structure", pdb_file)
    model = structure[0]

    token_chain_ids = []
    token_res_ids = []

    for chain in model:
        for residue in chain:
            if 'CA' in residue:
                token_chain_ids.append(chain.id)
                token_res_ids.append(residue.id[1])

    return token_chain_ids, token_res_ids


def parse_confidences_json(conf_path, pdb_path):
    with open(conf_path) as f:
        conf = json.load(f)

    chains = get_chains_from_pdb(pdb_path)
    pae = np.array(conf["pae"])
    token_chain_ids, token_res_ids = extract_token_chain_and_res_ids(pdb_path)

    chain_indices = {chain: [] for chain in chains}
    for i, chain in enumerate(token_chain_ids):
        chain_indices[chain].append(i)

    chain_pae = {
        chain: float(np.mean(pae[np.ix_(idxs, idxs)]))
        for chain, idxs in chain_indices.items()
    }

    ipae = {}
    pae_interaction = {}

    for ch1, ch2 in combinations(chains, 2):
        try:
            idx1_, idx2_ = get_interface_res_from_pdb(pdb_path, chain1=ch1, chain2=ch2)
            idx1 = [i for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                     if chain == ch1 and res_id in idx1_]
            idx2 = [i for i, (res_id, chain) in enumerate(zip(token_res_ids, token_chain_ids))
                     if chain == ch2 and res_id in idx2_]

            pair_key = f"{ch1}_{ch2}"

            if idx1 and idx2:
                ipae[pair_key] = np.mean([np.mean(pae[np.ix_(idx1, idx2)]), np.mean(pae[np.ix_(idx2, idx1)])])

            chain_1_indices = [i for i, chain in enumerate(token_chain_ids) if chain == ch1]
            chain_2_indices = [i for i, chain in enumerate(token_chain_ids) if chain == ch2]

            pae_interaction[pair_key] = np.mean([np.mean(pae[np.ix_(chain_1_indices, chain_2_indices)]),
                                                 np.mean(pae[np.ix_(chain_2_indices, chain_1_indices)])])

        except Exception as e:
            print(f"[Warning] Failed to process pair ({ch1}, {ch2}): {e}")

    return chain_pae, ipae, pae_interaction


def process_single_description(args):
    description, input_pdb_dir, base_dir = args
    try:
        base_path = Path(base_dir) / description / "seed-10_sample-0"
        summary_path = base_path / "summary_confidences.json"
        conf_path = base_path / "confidences.json"
        pdb_path = Path(input_pdb_dir) / f"{description}.pdb"

        if not summary_path.exists():
            return None, f"{description}: missing summary file"
        if not pdb_path.exists():
            return None, f"{description}: missing pdb file"
        if not conf_path.exists():
            return None, f"{description}: missing conf file"

        # Calculate ipSAE
        ipsae_metrics = {}
        pae_matrix, chain_ids, residue_types = load_tf_pae_and_chains(conf_path, pdb_path)
        ipsae_dict = calculate_ipsae(pae_matrix, chain_ids, residue_types, pae_cutoff=10)
        for k, v in ipsae_dict.items():
            ipsae_metrics[f"ipsae_{k}"] = v

        # Load confidence data
        summary = json.loads(summary_path.read_text())
        conf = json.loads(conf_path.read_text())
        chains = get_chains_from_pdb(pdb_path)

        # Chain-level ipTM and PTM
        iptm = dict(zip(chains, summary.get("chain_iptm", [])))
        ptm = dict(zip(chains, summary.get("chain_ptm", [])))

        # Inter-chain pair ipTM matrix
        iptm_matrix = summary["chain_pair_iptm"]
        interchain_iptm_dict = {}
        num_chains = len(chains)
        for i in range(num_chains):
            for j in range(i + 1, num_chains):
                interchain_iptm_dict[f"iptm_{chains[i]}_{chains[j]}"] = iptm_matrix[i][j]

        # pLDDT scores
        atom_plddts = conf["atom_plddts"]
        atom_chain_ids = conf["atom_chain_ids"]
        chain_plddt = {
            ch: float(np.mean([pl for pl, cid in zip(atom_plddts, atom_chain_ids) if cid == ch]))
            for ch in chains
        }

        # PAE-related metrics
        chain_pae, ipae, inter_pae = parse_confidences_json(conf_path, str(pdb_path))

        result = {
            "description": description,
            "ptm": summary.get("ptm", 0.0),
            "iptm": summary.get("iptm", 0.0),
        }

        for ch in chains:
            result[f"chain_{ch}_plddt"] = chain_plddt.get(ch, np.nan)
            result[f"chain_{ch}_pae"] = chain_pae.get(ch, np.nan)
            result[f"chain_{ch}_ptm"] = ptm.get(ch, np.nan)
            result[f"chain_{ch}_iptm"] = iptm.get(ch, np.nan)

        result.update(ipsae_metrics)
        result.update(interchain_iptm_dict)

        return result, None

    except Exception as e:
        return None, f"{description}: {str(e)}"


def extract_all_metrics_parallel(input_pdb_dir, base_dir, num_workers=None):
    descriptions = [d for d in os.listdir(base_dir) if (Path(base_dir) / d).is_dir()]
    args_list = [(d, input_pdb_dir, base_dir) for d in descriptions]

    results = []
    failed = []

    n_proc = num_workers or max(1, mp.cpu_count() - 1)
    with mp.Pool(processes=n_proc) as pool:
        for res, err in tqdm(pool.imap_unordered(process_single_description, args_list),
                             total=len(args_list), desc="Extracting metrics"):
            if err:
                failed.append(err)
            else:
                results.append(res)

    return pd.DataFrame(results), failed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_pdb_dir', required=True)
    parser.add_argument('--tfscore_output_dir', required=True)
    parser.add_argument('--save_metric_csv', required=True)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    df, failed = extract_all_metrics_parallel(args.input_pdb_dir, args.tfscore_output_dir, args.num_workers)
    df.to_csv(args.save_metric_csv, index=False)
    print(f"Done: {len(df)} succeeded, {len(failed)} failed")
    if failed:
        failed_path = Path(args.tfscore_output_dir) / "failed_records.txt"
        with open(failed_path, 'w') as fw:
            fw.write("\n".join(failed))
        print(f"Failed records written to {failed_path}")
