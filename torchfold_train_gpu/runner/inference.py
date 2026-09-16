"""torchfold.runner.inference
"""
from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# element / restype decode helpers
# --------------------------------------------------------------------------- #
def _z_to_symbol():
    """atomic number -> element symbol (via rdkit periodic table; fallback dict)."""
    try:
        from rdkit.Chem import GetPeriodicTable
        pt = GetPeriodicTable()
        return lambda z: (pt.GetElementSymbol(int(z)) if 1 <= int(z) <= 118 else "X")
    except Exception:
        common = {1: "H", 6: "C", 7: "N", 8: "O", 15: "P", 16: "S"}
        return lambda z: common.get(int(z), "X")


def _aatype_to_resname():
    """AF3-31-vocab restype index -> 3-letter (protein) / nucleotide name."""
    from torchfold.model.constants import residue_names as rn
    names = []
    for r in rn.POLYMER_TYPES_WITH_UNKNOWN_AND_GAP:  # 31 entries, AF3 order
        # protein 3-letter via one-letter map; nucleic stay as-is; gap/unk -> UNK/N
        if r in rn.PROTEIN_COMMON_ONE_TO_THREE:
            names.append(rn.PROTEIN_COMMON_ONE_TO_THREE[r])
        else:
            names.append(r if r != "-" else "UNK")
    table = {i: n for i, n in enumerate(names)}
    return lambda idx: table.get(int(idx), "UNK")


def _atom_name_from_chars(name_codes: np.ndarray) -> str:
    """[4] char codes (ord(c)-32) -> atom name string."""
    chars = [chr(int(c) + 32) for c in name_codes if int(c) > 0]
    return "".join(chars).strip() or "X"


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_model(device, af3_params_dir=None, ckpt=None, *, num_samples=5,
               num_recycles=10, diffusion_steps=200):
    from torchfold.model.alphafold3 import AlphaFold3
    from torchfold.weights.import_jax_weights import import_jax_weights_

    model = AlphaFold3(
        num_samples=num_samples, num_recycles=num_recycles,
        diffusion_steps=diffusion_steps, save_diffusion_debug=False,
        disable_internal_progress=True,
    ).eval()

    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        state = sd.get("model_state_dict", sd.get("model", sd))
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] {ckpt}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    else:
        import_jax_weights_(model, pathlib.Path(af3_params_dir))
        print(f"[params] loaded jax dump from {af3_params_dir}", flush=True)
    return model.to(device)


# --------------------------------------------------------------------------- #
# predict
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, batch, device):
    """Run one inference forward.

    Returns ``{"out": <raw model forward dict>, "coords": [N_sample, N_atom, 3]}``.

    ``out`` retains the confidence tensors the summary module needs
    (``pae_logits``, ``full_pde``, ``predicted_lddt``, ``distogram``). Coords are
    the dense diffusion samples flattened to per-atom layout (same dense->flat
    trick the summary module uses for plddt). Confidence / ranking is NOT computed
    here — ``confidence_summary.compute_full_data_and_summary`` owns that.
    """
    from torchfold.data.pipeline.adapter import standard_batch_to_af3
    from torchfold.runner.train import _dense_to_flat

    feats = batch["input_feature_dict"]
    a2t = feats["atom_to_token_idx"].to(torch.int64).to(device)
    a2ta = feats["atom_to_tokatom_idx"].to(torch.int64).to(device)

    af3 = standard_batch_to_af3(batch, device=device)
    out = model(af3)

    samples = out["diffusion_samples"]["atom_positions"].to(torch.float32)  # [S,T,D,3]
    coords = _dense_to_flat(samples, a2t, a2ta)                             # [S,N_atom,3]

    return {"out": out, "coords": coords}


# --------------------------------------------------------------------------- #
# build biotite AtomArray from the flat features
# --------------------------------------------------------------------------- #
def build_atom_array(batch):
    from biotite.structure import AtomArray

    feats = batch["input_feature_dict"]
    a2t = feats["atom_to_token_idx"].to(torch.int64).cpu().numpy()
    n_atom = int(a2t.shape[0])

    elem_z = feats["ref_element"].argmax(-1).cpu().numpy() + 1
    name_codes = feats["ref_atom_name_chars"].argmax(-1).cpu().numpy()  # [N_atom,4]
    restype_tok = feats["restype"].argmax(-1).cpu().numpy()           # [N_token] (32-vocab)
    resid_tok = feats["residue_index"].to(torch.int64).cpu().numpy()  # [N_token]
    asym_tok = feats["asym_id"].to(torch.int64).cpu().numpy()         # [N_token]
    is_lig_atom = feats["is_ligand"].to(torch.bool).cpu().numpy()     # [N_atom]

    from torchfold.data.pipeline.adapter import _restype_idx_32_to_31
    restype_af3 = _restype_idx_32_to_31(torch.from_numpy(restype_tok)).numpy()
    z2sym = _z_to_symbol()
    aa2res = _aatype_to_resname()

    def chain_letter(i):
        # 0->A ... 25->Z, 26->AA ...
        i = int(i); s = ""
        while True:
            s = chr(ord("A") + i % 26) + s; i = i // 26 - 1
            if i < 0:
                break
        return s

    arr = AtomArray(n_atom)
    arr.coord = np.zeros((n_atom, 3), dtype=np.float32)
    for i in range(n_atom):
        t = a2t[i]
        arr.atom_name[i] = _atom_name_from_chars(name_codes[i])
        arr.element[i] = z2sym(elem_z[i])
        arr.res_id[i] = int(resid_tok[t]) + 1
        arr.chain_id[i] = chain_letter(asym_tok[t])
        arr.res_name[i] = ("LIG" if is_lig_atom[i] else aa2res(restype_af3[t]))
        arr.hetero[i] = bool(is_lig_atom[i])
    return arr


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def build_test_dataloader(test_set, max_n_token=384, num_workers=4):
    from torchfold.configs.configs_data import data_configs
    from torchfold.config import parse_configs
    # Filter the test set to <= max_n_token AT THE DATASET LEVEL (only small/fast
    # structures are loaded) and use dataloader workers so the (potentially deep)
    # per-sample MSA featurization is parallelized/prefetched instead of blocking
    # the main thread with the GPU idle.
    base = {
        "seed": 42, "train_crop_size": 128, "test_max_n_token": max_n_token,
        "train_lig_atom_rename": False, "train_shuffle_mols": False,
        "train_shuffle_sym_ids": False, "test_lig_atom_rename": False,
        "test_shuffle_mols": False, "test_shuffle_sym_ids": False,
        "project": "x", "run_name": "x", "base_dir": "/tmp",
        "eval_interval": 1, "log_interval": 1, "max_steps": 1, "data": data_configs,
    }
    configs = parse_configs(base, fill_required_with_null=True)
    configs.data.num_dl_workers = num_workers
    configs.data.epoch_size = 16
    from torchfold.data.pipeline.dataloader import get_dataloaders
    _, test_dls = get_dataloaders(configs, world_size=1, seed=42)
    return test_dls[test_set]


def main():
    p = argparse.ArgumentParser(description="torchfold inference runner")
    p.add_argument("--af3_params_dir", default="./Alphafold3params",
                   help="JAX parameter dump directory (used unless --ckpt)")
    p.add_argument("--ckpt", default=None, help="torchfold {step}.pt; overrides the JAX dump")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--num_recycles", type=int, default=10)
    p.add_argument("--diffusion_steps", type=int, default=200)
    p.add_argument("--test_set", default="recentPDB_1536_sample384_0925")
    p.add_argument("--dataset_name", default=None,
                   help="output dataset dir name; default None -> use --test_set")
    p.add_argument("--num_targets", type=int, default=2)
    p.add_argument("--max_n_token", type=int, default=384)
    p.add_argument("--out_dir", default="./infer_out")
    p.add_argument("--num_seeds", type=int, default=1,
                   help="independent diffusion seeds per target; each seed = one "
                        "forward producing --num_samples samples -> one seed_{seed}/ dir")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--need_atom_confidence", action=argparse.BooleanOptionalAction,
                   default=False, help="also dump per-atom full_data JSON files")
    p.add_argument("--num_shards", type=int, default=1,
                   help="split accepted targets across this many parallel jobs")
    p.add_argument("--shard_idx", type=int, default=0,
                   help="which shard this job handles (0..num_shards-1)")
    args = p.parse_args()

    from torchfold.runner.confidence_summary import compute_full_data_and_summary
    from torchfold.runner.dumper import DataDumper

    dataset_name = args.dataset_name or args.test_set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device, af3_params_dir=(None if args.ckpt else args.af3_params_dir),
                       ckpt=args.ckpt, num_samples=args.num_samples,
                       num_recycles=args.num_recycles, diffusion_steps=args.diffusion_steps)
    print("[model] ready", flush=True)

    dumper = DataDumper(base_dir=args.out_dir,
                        need_atom_confidence=args.need_atom_confidence,
                        sorted_by_ranking_score=True)

    dl = build_test_dataloader(args.test_set, max_n_token=args.max_n_token, num_workers=4)
    done = 0          # global accepted targets (pre-shard), capped by --num_targets
    processed = 0     # targets actually predicted by THIS shard
    seeds_dumped = 0  # total (target, seed) dirs written by THIS shard
    for i, batch in enumerate(dl):
        feats = batch["input_feature_dict"]
        n_tok = int(feats["asym_id"].shape[0])
        if n_tok > args.max_n_token:
            continue
        if done >= args.num_targets:
            break
        accepted_idx = done
        done += 1
        # round-robin shard assignment over accepted targets
        if (accepted_idx % args.num_shards) != args.shard_idx:
            continue
        name = str(batch["basic"].get("pdb_id", f"target{i}"))
        n_atom = int(feats["atom_to_token_idx"].shape[0])
        print(f"\n==== predict {name} n_token={n_tok} n_atom={n_atom} "
              f"seeds={args.num_seeds}x samples={args.num_samples} "
              f"recycles={args.num_recycles} steps={args.diffusion_steps}", flush=True)

        atom_array = build_atom_array(batch)

        import time as _time
        # One forward per seed; each seed -> its own seed_{seed}/ dir + ranking.
        for s in range(args.num_seeds):
            seed = args.base_seed + s
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            pred = predict(model, batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize()

            summary_list, full_data_list = compute_full_data_and_summary(
                pred["out"], batch["input_feature_dict"],
                pred["coords"].to(device),
                num_recycles=args.num_recycles, need_full_data=True,
            )
            pred_dict = {
                "coordinate": pred["coords"],
                "summary_confidence": summary_list,
                "full_data": full_data_list,
            }
            dumper.dump(dataset_name, name, seed, pred_dict, atom_array)
            seeds_dumped += 1

            # best (rank0) summary -> concise log
            best = max(summary_list,
                       key=lambda d: d.get("ranking_score", float("-inf")))

            def _g(k):
                v = best.get(k, None)
                return (f"{float(v):.3f}" if isinstance(v, (int, float)) else "na")

            print(f"   seed {seed}: {pred['coords'].shape[0]} samples "
                  f"predict_s={(_time.perf_counter()-_t0):.3f} "
                  f"rank0 ranking_score={_g('ranking_score')} "
                  f"iptm={_g('iptm')} ptm={_g('ptm')} plddt={_g('plddt')}",
                  flush=True)
        processed += 1
    print(f"\nINFER_DONE shard {args.shard_idx}/{args.num_shards} "
          f"predicted {processed} target(s), dumped {seeds_dumped} seed-dir(s) "
          f"to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
