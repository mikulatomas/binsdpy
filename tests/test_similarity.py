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


@pytest.mark.parametrize(
    "similarity", [fun for _, fun in getmembers(similarity, isfunction)]
)
def test_similarity_mask(similarity):
    a_np = np.array([1, 1, 0, 0], dtype=bool)
    b_np = np.array([1, 0, 0, 1], dtype=bool)
    mask_np = np.array([1, 1, 0, 1], dtype=bool)
    a_np_masked = np.array([1, 1, 0], dtype=bool)
    b_np_masked = np.array([1, 0, 1], dtype=bool)

    Colors = bitset("Colors", ("red", "blue", "green", "yellow"))
    ColorsMasked = bitset("Colors", ("red", "blue", "yellow"))

    a_bitset = Colors.frommembers(["red", "blue"])
    b_bitset = Colors.frommembers(["red", "yellow"])
    mask_bitset = Colors.frommembers(["red", "blue", "yellow"])
    a_bitset_masked = ColorsMasked.frommembers(["red", "blue"])
    b_bitset_masked = ColorsMasked.frommembers(["red", "yellow"])

    assert similarity(a_np, b_np, mask=mask_np) == similarity(a_np_masked, b_np_masked)
    assert similarity(a_bitset, b_bitset, mask=mask_bitset) == similarity(
        a_bitset_masked, b_bitset_masked
    )
