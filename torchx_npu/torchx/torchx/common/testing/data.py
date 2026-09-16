"""Module that provides an abstraction for accessing test data."""

import os
import pathlib
from typing import Literal, overload

from absl.testing import absltest


class Data:
  """Provides an abstraction for accessing test data."""

  def __init__(self, data_dir: os.PathLike[str] | str):
    """Initiailizes data wrapper, providing users with high level data access.

    Args:
      data_dir: Directory containing test data.
    """
    self._data_dir = pathlib.Path(data_dir)

  def path(self, data_name: str | os.PathLike[str] | None = None) -> str:
    """Returns the path to a given test data.

    Args:
      data_name: the name of the test data file relative to data_dir. If not
        set, this will return the absolute path to the data directory.
    """
    data_dir_path = (
        pathlib.Path(absltest.get_default_test_srcdir()) / self._data_dir
    )

    if data_name:
      return str(data_dir_path / data_name)

    return str(data_dir_path)

  @overload
  def load(
      self, data_name: str | os.PathLike[str], mode: Literal['rt'] = 'rt'
  ) -> str:
    ...

  @overload
  def load(
      self, data_name: str | os.PathLike[str], mode: Literal['rb'] = 'rb'
  ) -> bytes:
    ...

  def load(
      self, data_name: str | os.PathLike[str], mode: str = 'rt'
  ) -> str | bytes:
    """Returns the contents of a given test data.

    Args:
      data_name: the name of the test data file relative to data_dir.
      mode: the mode in which to read the data file. Defaults to text ('rt').
    """
    with open(self.path(data_name), mode=mode) as f:
      return f.read()
