from typing import Any

import numpy as np
import pytest

import wax.external.eagerpy as ep


@pytest.mark.parametrize("fill_value", [0.0, 0, "0"])
def test_astensors_list_float(fill_value: Any) -> None:
    x = [
        fill_value,
    ] * 3
    ex = ep.astensors(x)
    assert isinstance(ex[0], ep.Tensor)
    ex_stacked = ep.stack(ex)
    x_stacked = ex_stacked.raw
    assert isinstance(x_stacked, np.ndarray)


def test_np_full_like() -> None:
    x, alpha = ep.astensors(
        np.array([1.10532079, 0.79016002, -1.45496991]), np.array(0.1)
    )
    x_like_alpha = ep.full_like(x, alpha)
    assert (x_like_alpha.raw == alpha.raw).all()
