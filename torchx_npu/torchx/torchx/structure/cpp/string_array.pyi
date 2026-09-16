

from collections.abc import Sequence
from typing import Any, overload

import numpy as np


def format_float_array(
    values: Sequence[float], num_decimal_places: int
) -> list[str]: ...


def isin(
    array: np.ndarray[object],
    test_elements: set[str | bytes],
    *,
    invert: bool = ...,
) -> np.ndarray[bool]: ...


@overload
def remap(
    array: np.ndarray[object],
    mapping: dict[str, str],
    default_value: str,
    inplace: bool = ...,
) -> np.ndarray[object]: ...


@overload
def remap(
    array: np.ndarray[object],
    mapping: dict[str, str],
    inplace: bool = ...,
) -> np.ndarray[object]: ...


def remap_multiple(
    arrays: Sequence[np.ndarray[object]],
    mapping: dict[tuple[Any], int],
) -> np.ndarray[int]: ...
