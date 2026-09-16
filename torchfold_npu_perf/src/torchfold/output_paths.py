import os
import pathlib


def safe_job_output_dir(
    output_root: os.PathLike[str] | str,
    job_name: str,
) -> str:
    """Return a direct child of output_root and reject path escapes."""
    root = pathlib.Path(output_root).resolve()
    candidate = pathlib.Path(output_root) / job_name
    if candidate.resolve().parent != root:
        raise ValueError(f"Unsafe output job name: {job_name!r}")
    return os.path.join(os.fspath(output_root), job_name)
