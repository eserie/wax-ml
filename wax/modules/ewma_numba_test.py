import numpy as np
import pandas as pd
import pytest

from wax.modules.ewma_numba import ewma, init


def test_ewma_numba():

    x = np.ones((30,), "float64")
    x[0] = np.nan

    x[1] = -1

    x[5:20] = np.nan
    x = x.reshape(-1, 1)
    state = init(x)
    res, state = ewma(com=10, adjust="linear")(x, state)
    pd.DataFrame(res).plot()


@pytest.mark.parametrize(
    "adjust, ignore_na",
    [(False, False), (False, True), (True, False), (True, True)],  # ,
)
def test_nan_at_beginning(adjust, ignore_na):

    T = 20
    x = np.full((T,), np.nan)
    x[2] = 1
    x[10] = -1

    compare_nan_at_beginning(x, com=10, adjust=adjust, ignore_na=ignore_na)

    # check min_periods option with random variable
    random_state = np.random.RandomState(42)
    x = random_state.normal(size=(5,))

    compare_nan_at_beginning(
        x,
        com=10,
        adjust=adjust,
        ignore_na=ignore_na,
        min_periods=2,
    )

    # check random variable with nans
    x = np.ones((30,), "float64")
    x[0] = np.nan
    x[1] = -1
    x[5:20] = np.nan

    compare_nan_at_beginning(
        x,
        com=10,
        adjust=adjust,
        ignore_na=ignore_na,
    )


def compare_nan_at_beginning(x, **ewma_kwargs):
    x = x.reshape(-1, 1)
    state = init(x)
    res, state = ewma(**ewma_kwargs)(x, state)
    res = pd.DataFrame(np.array(res))

    ref_res = pd.DataFrame(x).ewm(**ewma_kwargs).mean()
    pd.testing.assert_frame_equal(res, ref_res, atol=1.0e-6)
