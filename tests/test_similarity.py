from inspect import getmembers, isfunction

import pytest
import numpy as np
from bitsets import bitset

from binsdpy import similarity


@pytest.mark.parametrize(
    "similarity", [fun for _, fun in getmembers(similarity, isfunction)]
)
def test_similarity(similarity):
    a_np = np.array([1, 1, 0, 0], dtype=bool)
    b_np = np.array([1, 0, 0, 1], dtype=bool)

    Colors = bitset("Colors", ("red", "blue", "green", "yellow"))

    a_bitset = Colors.frommembers(["red", "blue"])
    b_bitset = Colors.frommembers(["red", "yellow"])

    assert similarity(a_np, b_np) == similarity(a_bitset, b_bitset)
