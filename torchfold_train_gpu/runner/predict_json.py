"""torchfold.runner.predict_json — JSON -> structure inference.

A AF3-faithful runner that drives the **JSON data path** (via
``torchfold.data.inference.get_inference_dataloader`` ->
``InferenceDataset``) instead of the curated test sets used by
``runner/inference.py``. It reuses ``runner/inference.py::load_model`` and
``::predict`` for the forward, ``runner/confidence_summary`` for the
confidence math, and ``runner/dumper.DataDumper`` for the torchfold output
layout. Optional MSA / template features are toggled per ``--use_msa`` /
``--use_template``; with both disabled the run does NOT require any template
mmCIF / MSA database directories.

Output layout (mirrors runner/inference.py):
    {out_dir}/{dataset_name}/{sample_name}/seed_{seed}/predictions/
        {sample_name}_sample_{rank}.cif
        {sample_name}_summary_confidence_sample_{rank}.json
        {sample_name}_full_data_sample_{rank}.json   (only if --need_atom_confidence)

Usage (1 GPU, project python env):
    python -m torchfold.runner.predict_json \
        --input_json /path/to/inputs.json \
        --out_dir ./infer_json_out --dataset_name json \
        --num_samples 5 --num_recycles 10 --diffusion_steps 200 \
        --num_seeds 1 --base_seed 42
    # with MSA/template:
        --use_msa --use_template
    # or load a trained checkpoint instead of raw AF3 params:
        --ckpt /.../checkpoints/500.pt
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# config assembly for get_inference_dataloader
# --------------------------------------------------------------------------- #
def build_inference_configs(args):
    """Build a ConfigDict suitable for ``get_inference_dataloader``.

    Mirrors ``runner/inference.py::build_test_dataloader``'s
    ``parse_configs(base, fill_required_with_null=True)`` pattern, then sets
    the inference-data-path fields that ``InferenceDataset`` reads
    (``input_json_path`` / ``use_msa`` / ``use_template`` / ``num_workers`` /
    ``dump_dir`` / ``esm.*`` / ``sample_diffusion.guidance.enable`` /
    ``load_checkpoint_dir``). Fields that don't exist in the base config are
    added post-parse (ml_collections ``ConfigDict`` allows adding new keys via
    attribute assignment).
    """
    from torchfold.configs.configs_base import configs as base_configs
    from torchfold.configs.configs_data import data_configs
    from torchfold.config import parse_configs

    # mirror build_test_dataloader: fill RequiredValue(...) basics so parse
    # succeeds, attach the rich data.<...> block from configs_data, and keep the
    # assembled base (model / sample_diffusion / loss / optim) so nested fields
    # like sample_diffusion.* parse correctly.
    base = dict(base_configs)
    base.update(
        {
            "seed": args.base_seed,
            "project": "x",
            "run_name": "x",
            "base_dir": "/tmp",
            "eval_interval": 1,
            "log_interval": 1,
            "data": data_configs,
        }
    )
    # test_max_n_token only matters when a token cap is requested.
    if args.max_n_token is not None:
        base["data"]["test_max_n_token"] = args.max_n_token

    configs = parse_configs(base, fill_required_with_null=True)

    # ---- inference data-path fields read by InferenceDataset ----
    configs.input_json_path = args.input_json
    # InferenceDataset.__init__ reads configs.dump_dir (stored, otherwise
    # unused by the no-MSA/no-template path); point it at the output dir.
    configs.dump_dir = args.out_dir
    configs.use_msa = bool(args.use_msa)
    configs.use_template = bool(args.use_template)
    configs.num_workers = int(args.num_workers)
    # InferenceDataset reads these with .get(...) defaults; set explicitly so
    # behaviour is deterministic regardless of base-config presence.
    configs.msa_pair_as_unpair = True
    configs.use_rna_msa = True
    # load_checkpoint_dir is used by ESMFeaturizer.precompute_esm_embedding (only
    # when esm.enable); point it at the AF3 params dir as a safe default.
    configs.load_checkpoint_dir = args.af3_params_dir or ""

    # ---- ESM: keep disabled. InferenceDataset unconditionally writes
    # configs.esm.embedding_dir / .sequence_fpath and reads .model_name even
    # when disabled, so the esm subtree must exist with those fields. ----
    if "esm" not in configs:
        configs.esm = {}
    configs.esm.enable = False
    # subfields referenced even with enable=False:
    if "model_name" not in configs.esm:
        configs.esm.model_name = "esm2_3B"  # NOTE: only used to build the
        # embedding_dir path string when ESM is disabled; any non-empty name is
        # fine (confirmed: both reference runs ran with esm.enable=False).
    if "embedding_dim" not in configs.esm:
        configs.esm.embedding_dim = 1028
    if "embedding_dir" not in configs.esm:
        configs.esm.embedding_dir = ""
    if "sequence_fpath" not in configs.esm:
        configs.esm.sequence_fpath = ""

    # ---- diffusion guidance: InferenceDataset reads
    # configs.sample_diffusion.guidance.enable (controls TFG feature extraction).
    # Keep disabled; add the subtree if missing. ----
    if "sample_diffusion" not in configs:
        configs.sample_diffusion = {}
    if "guidance" not in configs.sample_diffusion:
        configs.sample_diffusion.guidance = {}
    configs.sample_diffusion.guidance.enable = False

    # ---- template: with use_template the dataset needs data.template.* dirs.
    # With --no-use_template (default) we must NOT require any mmcif dir -- the
    # dataset only touches data.template.* when use_template is True
    # (infer_dataloader.py gates on configs.use_template). Nothing to set for the
    # no-template path. ----
    if args.use_template:
        if "template" not in configs.data:
            configs.data.template = {}
        # The base config (configs_data.template) points prot_template_mmcif_dir
        # at an empty dir (torchfold/mmcif). Redirect it to the populated PDB
        # mmCIF library so template structures resolve from local disk.
        if getattr(args, "template_mmcif_dir", None):
            configs.data.template.prot_template_mmcif_dir = args.template_mmcif_dir
        # Enforce OFFLINE template resolution: without this, fetch_remote defaults
        # to True (configs_data has no fetch_remote key) and a missing cif would
        # silently trigger a PDBe HTTP download. With fetch_remote=False the
        # featurizer reads cifs only from prot_template_mmcif_dir; a missing cif is
        # skipped per-hit (template_utils.py:702), never a network call. This also
        # arms the local-mmcif-exists assertion in infer_dataloader.py.
        configs.data.template.fetch_remote = False
    return configs


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="torchfold AF3 JSON->structure inference (torchfold output)"
    )
    p.add_argument("--input_json", required=True,
                   help="torchfold-format input JSON (list of samples with 'name'/'sequences')")
    p.add_argument("--out_dir", default="./infer_json_out")
    p.add_argument("--dataset_name", default="json",
                   help="output dataset dir name")
    p.add_argument("--use_msa", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--use_template", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--af3_params_dir",
                   default="./Alphafold3params",
                   help="JAX parameter dump directory (used unless --ckpt)")
    p.add_argument("--ckpt", default=None, help="torchfold {step}.pt; overrides the JAX dump")
    p.add_argument("--num_samples", type=int, default=5)
    p.add_argument("--num_seeds", type=int, default=1,
                   help="independent diffusion seeds per sample; each seed = one "
                        "forward producing --num_samples samples -> one seed_{seed}/ dir")
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--num_recycles", type=int, default=10)
    p.add_argument("--diffusion_steps", type=int, default=200)
    p.add_argument("--need_atom_confidence", action=argparse.BooleanOptionalAction,
                   default=False, help="also dump per-atom full_data JSON files")
    p.add_argument("--max_n_token", type=int, default=None,
                   help="optional token cap; samples above it are skipped")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--template_mmcif_dir",
                   default=os.environ.get("MMCIF_DIR", ""),
                   help="populated PDB mmCIF library used to resolve template "
                        "structures (offline); only used with --use_template. "
                        "Default: $MMCIF_DIR or empty")
    args = p.parse_args()

    from torchfold.runner.inference import load_model, predict
    from torchfold.runner.confidence_summary import compute_full_data_and_summary
    from torchfold.runner.dumper import DataDumper
    from torchfold.data.inference import get_inference_dataloader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        device,
        af3_params_dir=(None if args.ckpt else args.af3_params_dir),
        ckpt=args.ckpt,
        num_samples=args.num_samples,
        num_recycles=args.num_recycles,
        diffusion_steps=args.diffusion_steps,
    )
    print("[model] ready", flush=True)

    configs = build_inference_configs(args)
    dl = get_inference_dataloader(configs)

    dumper = DataDumper(
        base_dir=args.out_dir,
        need_atom_confidence=args.need_atom_confidence,
        sorted_by_ranking_score=True,
    )

    import time as _time

    n_done = 0
    seeds_dumped = 0
    for item in dl:
        # batch_size=1 + collate_fn_identity -> the DataLoader returns the raw
        # __getitem__ tuple wrapped in a length-1 list. InferenceDataset
        # returns (data, atom_array, error_message).
        # collate_fn_identity wraps the __getitem__ tuple in a 1-elem batch
        # list (confirmed by both reference runs); unwrap it here.
        if isinstance(item, (list, tuple)) and len(item) == 1:
            item = item[0]
        data, atom_array, error_message = item

        # name lives on data["sample_name"] (set by InferenceDataset.__getitem__).
        name = str(data.get("sample_name", f"sample{data.get('sample_index', n_done)}"))

        if error_message:
            print(f"[skip] {name}: featurization failed: {error_message}", flush=True)
            continue
        if atom_array is None or "input_feature_dict" not in data:
            print(f"[skip] {name}: empty data / no input_feature_dict", flush=True)
            continue

        feats = data["input_feature_dict"]
        n_tok = int(feats["asym_id"].shape[0])
        n_atom = int(feats["atom_to_token_idx"].shape[0])
        if args.max_n_token is not None and n_tok > args.max_n_token:
            print(f"[skip] {name}: n_token={n_tok} > max_n_token={args.max_n_token}", flush=True)
            continue

        print(f"\n==== predict {name} n_token={n_tok} n_atom={n_atom} "
              f"seeds={args.num_seeds}x samples={args.num_samples} "
              f"recycles={args.num_recycles} steps={args.diffusion_steps}", flush=True)

        for s in range(args.num_seeds):
            seed = args.base_seed + s
            torch.manual_seed(seed)
            np.random.seed(seed % (2 ** 32))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            pred = predict(model, data, device)
            if device.type == "cuda":
                torch.cuda.synchronize()

            summary_list, full_data_list = compute_full_data_and_summary(
                pred["out"], data["input_feature_dict"],
                pred["coords"].to(device),
                num_recycles=args.num_recycles, need_full_data=True,
            )
            pred_dict = {
                "coordinate": pred["coords"],
                "summary_confidence": summary_list,
                "full_data": full_data_list,
            }
            dumper.dump(args.dataset_name, name, seed, pred_dict, atom_array)
            seeds_dumped += 1

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
        n_done += 1

    print(f"\nINFER_DONE predicted {n_done} sample(s), dumped {seeds_dumped} "
          f"seed-dir(s) to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
