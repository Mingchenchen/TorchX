from __future__ import annotations


def balanced_partition_range(
    total_size: int,
    num_parts: int,
    part_index: int,
) -> tuple[int, int]:
    """Return one non-empty, contiguous slice of a balanced partition."""
    if total_size < 1:
        raise ValueError(f"total_size must be positive, got {total_size}")
    if num_parts < 1:
        raise ValueError(f"num_parts must be positive, got {num_parts}")
    if num_parts > total_size:
        raise ValueError(
            f"num_parts={num_parts} cannot exceed total_size={total_size}"
        )
    if part_index < 0 or part_index >= num_parts:
        raise ValueError(
            f"part_index={part_index} must be in [0, {num_parts})"
        )

    start = total_size * part_index // num_parts
    end = total_size * (part_index + 1) // num_parts
    return start, end
