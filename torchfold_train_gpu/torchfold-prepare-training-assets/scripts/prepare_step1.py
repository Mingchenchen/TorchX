#!/usr/bin/env python3
"""Run Step1: CIF -> bioassembly PKL + indices.csv.

Requires the `protenix` package (used as the structure preprocessor) on
PYTHONPATH, and CCD files under $PROTENIX_ROOT_DIR/common/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    from joblib import Parallel, delayed
    from tqdm import tqdm
except ImportError as exc:
    raise SystemExit(
        "Step1 needs pandas, joblib, and tqdm (provided by the protenix install).\n"
        f"Import error: {exc}"
    ) from exc

try:
    from protenix.data.pipeline.data_pipeline import DataPipeline
    from protenix.utils.file_io import dump_gzip_pickle
except ImportError as exc:
    raise SystemExit(
        "Cannot import protenix. Install it first:\n"
        "  pip install protenix\n"
        "or clone https://github.com/bytedance/Protenix and set:\n"
        "  export PYTHONPATH=/path/to/Protenix:$PYTHONPATH\n"
        f"Import error: {exc}"
    ) from exc


def gen_a_bioassembly_data(
    mmcif: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
):
    dataset = "Distillation" if distillation else "WeightedPDB"
    sample_indices_list, bioassembly_dict = DataPipeline.get_data_from_mmcif(
        mmcif, cluster_file, dataset
    )
    if sample_indices_list and bioassembly_dict:
        sample_id = bioassembly_dict["pdb_id"]
        dump_gzip_pickle(
            bioassembly_dict, bioassembly_output_dir / f"{sample_id}.pkl.gz"
        )
        return sample_indices_list
    return None


def run_gen_data(
    input_path: Path,
    output_indices_csv: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
    distillation: bool = False,
    num_workers: int = 1,
) -> None:
    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_indices_csv = Path(output_indices_csv)
    output_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        mmcif_list = list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
    elif input_path.suffix == ".txt":
        with open(input_path) as f:
            mmcif_list = [Path(i.strip()) for i in f.readlines() if i.strip() and not i.strip().startswith("#")]
    else:
        raise SystemExit(f"Unsupported input path: {input_path}")

    if not mmcif_list:
        raise SystemExit(f"No CIF paths found in {input_path}")

    all_sample_indices_list = [
        r
        for r in tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
                delayed(gen_a_bioassembly_data)(
                    mmcif, bioassembly_output_dir, cluster_file, distillation
                )
                for mmcif in mmcif_list
            ),
            total=len(mmcif_list),
        )
    ]

    merged_results = []
    for sample_indices_list in all_sample_indices_list:
        if sample_indices_list:
            merged_results += sample_indices_list
    if not merged_results:
        raise SystemExit("Step1 produced no indices (all CIFs failed). Check Step1 logs.")
    df = pd.DataFrame(merged_results)
    df.to_csv(output_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input_path", type=Path, required=True)
    parser.add_argument("-o", "--output_csv", type=Path, required=True)
    parser.add_argument("-b", "--bio_output_dir", type=Path, required=True)
    parser.add_argument("-c", "--cluster_file", type=Path, default=None)
    parser.add_argument(
        "-d",
        "--distillation",
        action="store_true",
        help="Use Distillation parser (predicted / minimal CIFs). Default off.",
    )
    parser.add_argument("-n", "--n_cpu", type=int, default=1)
    args = parser.parse_args()

    root_dir = os.environ.get("PROTENIX_ROOT_DIR", "")
    if not root_dir:
        print(
            "[WARN] PROTENIX_ROOT_DIR is unset; Protenix will look for CCD under $HOME/common/",
            file=sys.stderr,
        )
    run_gen_data(
        input_path=args.input_path,
        output_indices_csv=args.output_csv,
        bioassembly_output_dir=args.bio_output_dir,
        cluster_file=args.cluster_file,
        distillation=args.distillation,
        num_workers=max(1, args.n_cpu),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
