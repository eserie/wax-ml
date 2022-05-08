# Copyright 2021 The WAX-ML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pandas as pd
import pytest

numba = pytest.importorskip("numba")


def test_ewma_numba():
    from wax.numba.ewma_numba import ewma

    x = np.ones((30,), "float64")
    x[0] = np.nan

    x[1] = -1

    x[5:20] = np.nan
    x = x.reshape(-1, 1)
    res, state = ewma(com=10, adjust="linear")(x)
    pd.DataFrame(res).plot()


def check_against_pandas_ewm(x, **ewma_kwargs):
    from wax.numba.ewma_numba import ewma

    res, state = ewma(**ewma_kwargs)(x, state=None)
    res = pd.DataFrame(np.array(res))

    ref_res = pd.DataFrame(x).ewm(**ewma_kwargs).mean()
    pd.testing.assert_frame_equal(res, ref_res, atol=1.0e-6)
    return res


@pytest.mark.parametrize(
    "adjust, ignore_na",
    [(False, False), (False, True), (True, False), (True, True)],  # ,
)
def test_nan_at_beginning(adjust, ignore_na):
    T = 20
    x = np.full((T,), np.nan)
    x[2] = 1
    x[10] = -1

    check_against_pandas_ewm(x, com=10, adjust=adjust, ignore_na=ignore_na)

    # check min_periods option with random variable
    random_state = np.random.RandomState(42)
    x = random_state.normal(size=(5,))
    check_against_pandas_ewm(
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
    check_against_pandas_ewm(
        x,
        com=10,
        adjust=adjust,
        ignore_na=ignore_na,
    )


def test_init_value():
    from wax.numba.ewma_numba import ewma

    # check random variable with nans
    x = np.ones((30,), "float64")
    x[0] = np.nan
    x[1] = -1
    x[5:20] = np.nan

    x = x.reshape(-1, 1)
    res, state = ewma(com=10, adjust=False, ignore_na=False)(x)

    res_init0, state = ewma(com=10, adjust=False, ignore_na=False, initial_value=0)(x)

    assert res_init0[0] == 0
    assert np.isnan(res[0])
    assert np.linalg.norm(res_init0) < np.linalg.norm(np.nan_to_num(res))


def test_ewma_state():
    from wax.numba.ewma_numba import ewma

    x = np.ones((30,), "float64")
    x[0] = np.nan
    x[1] = -1
    x[5:20] = np.nan

    x = x.reshape(-1, 1)

    ewma_apply = ewma(com=10, adjust="linear", min_periods=5)
    res_full, _ = ewma_apply(x)

    T = 10
    res1, state = ewma_apply(x[:T])
    res2, _ = ewma_apply(x[T:], state)
    res12 = np.concatenate([res1, res2])
    df = pd.concat(
        [pd.DataFrame(res_full), pd.DataFrame(res12)], axis=1, keys=["full", "12"]
    )
    assert len(df)

    assert np.allclose(res_full, res12, equal_nan=True)


@pytest.mark.parametrize("obj_type", ["frame", "series"])
def test_pandas_online(obj_type):
    from wax.numba.ewma_numba import register_wax_numba

    x = np.ones((30,), "float64")
    x[0] = np.nan
    x[1] = -1
    x[5:20] = np.nan

    if obj_type == "frame":
        x = x.reshape(-1, 1)
        X = pd.DataFrame(x)
    else:
        X = pd.Series(x)

    register_wax_numba()
    res_full = X.wax_numba.ewm(com=10).mean()
    res1, state = X.iloc[:10].wax_numba.ewm(com=10, return_state=True).mean()
    res2 = X.iloc[10:].wax_numba.ewm(com=10).mean(state=state)

    res12 = pd.concat([res1, res2])
    if obj_type == "frame":
        pd.testing.assert_frame_equal(res_full, res12)
    else:
        pd.testing.assert_series_equal(res_full, res12)

    assert state.dtypes.tolist() == [np.float64, np.float64, np.int64]
