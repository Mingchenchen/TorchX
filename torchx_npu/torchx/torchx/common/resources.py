"""Load external resources, such as external tools or data resources."""

import os
import pathlib
import typing
from collections.abc import Iterator
from importlib import resources
from typing import BinaryIO, Final, Literal, TextIO

import torchx.common

_DATA_ROOT:  Final[pathlib.Path] = (
    resources.files(torchx.common).joinpath('..').resolve()
)
ROOT = _DATA_ROOT


def filename(name: str | os.PathLike[str]) -> str:
  """Returns the absolute path to an external resource.

  Note that this calls resources.GetResourceFilename under the hood and hence
  causes par file unpacking, which might be unfriendly on diskless machines.


  Args:
    name: the name of the resource corresponding to its path relative to the
      root of the repository.
  """
  return (_DATA_ROOT / name).as_posix()


@typing.overload
def open_resource(
    name: str | os.PathLike[str], mode: Literal['r', 'rt'] = 'rt'
) -> TextIO:
  ...


@typing.overload
def open_resource(
    name: str | os.PathLike[str], mode: Literal['rb']
) -> BinaryIO:
  ...


def open_resource(
    name: str | os.PathLike[str], mode: str = 'rb'
) -> TextIO | BinaryIO:
  """Returns an open file object for the named resource.

  Args:
    name: the name of the resource corresponding to its path relative to the
      root of the repository.
    mode: the mode to use when opening the file.
  """
  return (_DATA_ROOT / name).open(mode)


def get_resource_dir(path: str | os.PathLike[str]) -> os.PathLike[str]:
  return _DATA_ROOT / path


def walk(path: str) -> Iterator[tuple[str, list[str], list[str]]]:
  """Walks the directory tree of resources similar to os.walk."""
  return os.walk((_DATA_ROOT / path).as_posix())
