

"""Interface and implementations for fetching MSA data."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from torchx.data import msa
from torchx.data import msa_config

MsaErrors: TypeAlias = Sequence[tuple[msa_config.RunConfig, str]]


class MsaProvider(Protocol):
  """Interface for providing Multiple Sequence Alignments."""

  def __call__(
      self,
      query_sequence: str,
      chain_polymer_type: str,
  ) -> tuple[msa.Msa, MsaErrors]:
    """Retrieve MSA for the given polymer query_sequence.

    Args:
      query_sequence: The residue sequence of the polymer to search for.
      chain_polymer_type: The polymer type of the query_sequence. This must
        match the chain_polymer_type of the provider.

    Returns:
      A tuple containing the MSA and MsaErrors. MsaErrors is a Sequence
      containing a tuple for each msa_query that failed. Each tuple contains
      the failing query and the associated error message.
    """


class EmptyMsaProvider:
  """MSA provider that returns just the query sequence, useful for testing."""

  def __init__(self, chain_polymer_type: str):
    self._chain_polymer_type = chain_polymer_type

  def __call__(
      self, query_sequence: str, chain_polymer_type: str
  ) -> tuple[msa.Msa, MsaErrors]:
    """Returns an MSA containing just the query sequence, never errors."""
    if chain_polymer_type != self._chain_polymer_type:
      raise ValueError(
          f'EmptyMsaProvider of type {self._chain_polymer_type} called with '
          f'sequence of {chain_polymer_type=}, {query_sequence=}.'
      )
    return (
        msa.Msa.from_empty(
            query_sequence=query_sequence,
            chain_poly_type=self._chain_polymer_type,
        ),
        (),
    )
