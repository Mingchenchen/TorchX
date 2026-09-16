

"""Interface and implementations for fetching templates data."""

import datetime
from collections.abc import Mapping
from typing import Any, Protocol, TypeAlias

TemplateFeatures: TypeAlias = Mapping[str, Any]


class TemplateFeatureProvider(Protocol):
  """Interface for providing Template Features."""

  def __call__(
      self,
      sequence: str,
      release_date: datetime.date | None,
      include_ligand_features: bool = True,
  ) -> TemplateFeatures:
    """Retrieve template features for the given sequence and release_date.

    Args:
      sequence: The residue sequence of the query.
      release_date: The release_date of the template query, this is used to
        filter templates for training, ensuring that they do not leak structure
        information from the future.
      include_ligand_features: Whether to include ligand features.

    Returns:
      Template features: A mapping of template feature labels to features, which
        may be numpy arrays, bytes objects, or for the special case of label
        `ligand_features`, a nested feature map of labels to numpy arrays.

    Raises:
      TemplateRetrievalError if the template features were not found.
    """
