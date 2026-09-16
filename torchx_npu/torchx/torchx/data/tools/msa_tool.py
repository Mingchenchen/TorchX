

"""Defines protocol for MSA tools."""

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MsaToolResult:
  """The result of a MSA tool query."""

  target_sequence: str
  e_value: float
  a3m: str


class MsaTool(Protocol):
  """Interface for MSA tools."""

  def query(self, target_sequence: str) -> MsaToolResult:
    """Runs the MSA tool on the target sequence."""
