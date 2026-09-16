

"""Type annotations for Python bindings for `msa_conversion`.

The type annotations in this file were modified from the automatically generated
stubgen output.
"""

from collections.abc import Iterable


def align_sequence_to_gapless_query(
    sequence: str | bytes,
    query_sequence: str | bytes,
) -> str: ...


def convert_a3m_to_stockholm(a3m_sequences: Iterable[str]) -> list[str]: ...
