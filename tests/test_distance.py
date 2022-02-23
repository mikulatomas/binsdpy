from inspect import getmembers, isfunction

import pytest
import numpy as np
from bitsets import bitset

from binsdpy import distance


@pytest.mark.parametrize(
    "distance", [fun for _, fun in getmembers(distance, isfunction)]
)
def test_distance(distance):
    a_np = np.array([1, 1, 0, 0], dtype=bool)
    b_np = np.array([1, 0, 0, 1], dtype=bool)

    Colors = bitset("Colors", ("red", "blue", "green", "yellow"))

    a_bitset = Colors.frommembers(["red", "blue"])
    b_bitset = Colors.frommembers(["red", "yellow"])

    assert distance(a_np, b_np) == distance(a_bitset, b_bitset)
