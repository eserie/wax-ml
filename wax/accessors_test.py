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
import numpy as onp
import pandas as pd
import pytest
import xarray as xr
from jax.config import config

from wax.accessors import register_wax_accessors
from wax.datasets.generate_temperature_data import generate_temperature_data
from wax.modules import EWMA
from wax.stream_test import prepare_test_data
from wax.testing import assert_tree_all_close


def module_same_shape(x):
    return EWMA(1.0 / 10.0)(x)


def module_reduce_shape(x):
    return EWMA(1.0 / 10.0)(x).sum()


def module_map(x):
    return {"ts_10": EWMA(1.0 / 10.0)(x).sum(), "ts_5": EWMA(1.0 / 5.0)(x).sum()}


def check_ema_state(state, ref_count=124):
    assert (state["ewma"]["count"] == ref_count).all()


def prepare_format_data(format):
    dataset = prepare_test_data()

    if format == "dataset":
        return dataset
    data = dataset["ground"]

    if format == "dataframe":
        # simple index dataframe
        # data = data.isel(lon=0).to_pandas()
        data = data.to_series().unstack()
    elif format == "series":
        data = data.to_series()
    else:
        assert format == "dataarray"
    return data


def test_unroll_dataset_accessor():
    config.update("jax_enable_x64", False)
    register_wax_accessors()

    dataset = prepare_format_data("dataset")

    def module(dataset):
        return EWMA(1.0 / 10.0)(dataset["ground"])

    with pytest.raises(ValueError):
        dataset.wax.stream().apply(
            module, format_dims=onp.array(["time", "lon", "lat"])
        )

    xoutput, state = dataset.wax.stream(
        local_time="time",
        freqs={"day": "d"},
        ffills={"day": True},
        buffer_maxlen={"news_time": 3},
        verbose=False,
        pbar=False,
    ).apply(module, format_dims=onp.array(["time", "lon", "lat"]))
    assert xoutput.shape == (124, 53, 25)
    assert isinstance(xoutput, xr.DataArray)
    check_ema_state(state)


@pytest.mark.parametrize("format", ["dataarray", "dataframe", "series"])
def test_unroll_dataarray_accessor_module_same_shape(format):
    config.update("jax_enable_x64", False)
    module = module_same_shape
    register_wax_accessors()
    data = prepare_format_data(format)
    if format == "dataframe":
        assert data.index.nlevels == 2
        assert data.columns.nlevels == 1
    xoutput, state = data.wax.stream().apply(module)
    assert type(xoutput) == type(data)
    assert xoutput.shape == data.shape
    check_ema_state(state)


@pytest.mark.parametrize("format", ["dataframe"])
def test_unroll_dataframe_transposed_accessor_module(format):
    config.update("jax_enable_x64", False)
    module = module_same_shape
    register_wax_accessors()
    data = prepare_format_data(format).unstack()
    if format == "dataframe":
        assert data.index.nlevels == 1
        assert data.columns.nlevels == 2
    xoutput, state = data.wax.stream().apply(module)
    assert type(xoutput) == type(data)
    assert xoutput.shape == data.shape  # (124, 53, 25)
    check_ema_state(state)


def test_unroll_dataframe_accessor_simple_index():
    config.update("jax_enable_x64", False)
    module = module_same_shape
    register_wax_accessors()
    data = prepare_format_data("dataarray")
    data = data.isel(lon=0).to_pandas()
    assert data.index.nlevels == 1
    assert data.columns.nlevels == 1

    xoutput, state = data.wax.stream().apply(module)

    assert type(xoutput) == type(data)

    assert xoutput.shape == data.shape
    check_ema_state(state)


@pytest.mark.parametrize("format", ["dataarray", "dataframe", "series"])
def test_unroll_dataarray_accessor_module_reduce_shape(format):
    config.update("jax_enable_x64", False)
    module = module_reduce_shape
    register_wax_accessors()
    data = prepare_format_data(format)
    xoutput, state = data.wax.stream().apply(module)

    if format in ["series", "dataarray"]:
        assert type(xoutput) == type(data)
    else:
        assert isinstance(xoutput, pd.Series)

    assert xoutput.shape == (124,)
    check_ema_state(state)


@pytest.mark.parametrize("format", ["dataarray", "dataframe", "series"])
def test_unroll_dataarray_accessor_module_map(format):
    config.update("jax_enable_x64", False)
    module = module_map
    register_wax_accessors()
    data = prepare_format_data(format)
    xoutput, state = data.wax.stream().apply(module)

    assert isinstance(xoutput, dict)
    assert xoutput.keys() == {"ts_10", "ts_5"}

    check_ema_state(state, ref_count=124)


def test_unroll_dataset_accessor_module_map():
    config.update("jax_enable_x64", False)

    def module_map(dataset):
        return {
            name: {
                "ts_10": EWMA(1.0 / 10.0, name=name)(val).sum(),
                "ts_5": EWMA(1.0 / 5.0, name=name)(val).sum(),
            }
            for name, val in dataset.items()
            if name in ["ground", "air"]
        }

    module = module_map
    register_wax_accessors()
    dataset = prepare_test_data()
    xoutput, state = dataset.wax.stream(local_time="time").apply(
        module, format_dims=["time"]
    )
    assert isinstance(xoutput, dict)
    assert xoutput.keys() == {"air", "ground"}
    assert state.keys() == {"air", "air_1", "ground", "ground_1"}
    check_ema_state({"ewma": state["ground"]}, ref_count=124)
    check_ema_state({"ewma": state["ground_1"]}, ref_count=124)
    check_ema_state({"ewma": state["air"]}, ref_count=6)
    check_ema_state({"ewma": state["air_1"]}, ref_count=6)


@pytest.mark.parametrize("format", ["dataarray", "dataframe", "series"])
def test_wax_ewma(format):
    config.update("jax_enable_x64", False)
    register_wax_accessors()
    data = prepare_format_data(format)
    ema, state = data.wax.ewm(alpha=0.1, adjust=True, return_state=True).mean()
    check_ema_state(state, ref_count=124)


def _compute_ewma_direct(dataarray):
    import haiku as hk
    import jax.numpy as jnp

    from wax.unroll import dynamic_unroll

    seq = hk.PRNGSequence(42)
    x = jnp.array(dataarray.values, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust=True)(x)

    ema3, state3 = dynamic_unroll(model, None, None, next(seq), False, x)
    return ema3, state3


@pytest.mark.parametrize("format", ["dataframe"])
def test_wax_ewma_vs_pandas(format):

    config.update("jax_enable_x64", True)
    register_wax_accessors()
    data = prepare_format_data(format)

    pandas_ema = data.unstack().ewm(alpha=0.1, adjust=True).mean().stack()
    ema = data.wax.ewm(alpha=0.1, adjust=True).mean()
    ema2, state = data.wax.ewm(alpha=0.1, adjust=True, return_state=True).mean()

    du = data.unstack()
    ema3, state3 = _compute_ewma_direct(du)
    ema3 = pd.DataFrame(ema3, index=du.index, columns=du.columns).stack().values

    assert onp.allclose(ema2, pandas_ema.values)
    assert onp.allclose(ema3, pandas_ema.values)
    assert onp.allclose(ema3, ema.values)
    check_ema_state(state, ref_count=124)


def test_air_temperature_dataset():
    config.update("jax_enable_x64", True)
    register_wax_accessors()
    # da = xr.tutorial.open_dataset("air_temperature")
    da = generate_temperature_data()

    def diff(y):
        return y["air"].flatten().sum()

    output = da.wax.stream(return_state=False).apply(diff, format_dims=da.air.dims)
    check = da.air.to_series().sum(level=0)
    assert_tree_all_close(output.to_series(), check)


def test_air_temperature_dataarray():
    config.update("jax_enable_x64", True)
    register_wax_accessors()
    # da = xr.tutorial.open_dataset("air_temperature")
    da = generate_temperature_data()
    da = da.air

    def diff(y):
        return y.flatten().sum()

    output = da.wax.stream(return_state=False).apply(diff, format_dims=da.dims)
    check = da.to_series().sum(level=0)
    assert_tree_all_close(output.to_series(), check)


def test_air_temperature_dataframe():
    config.update("jax_enable_x64", True)
    register_wax_accessors()
    # da = xr.tutorial.open_dataset("air_temperature")
    da = generate_temperature_data()
    da = da.air.isel(lat=0)

    def diff(y):
        return y.flatten().sum()

    output = (
        da.to_pandas().wax.stream(return_state=False).apply(diff, format_dims=da.dims)
    )
    check = da.to_series().sum(level=0)
    assert_tree_all_close(output, check)


def test_air_temperature_series():
    config.update("jax_enable_x64", True)
    register_wax_accessors()
    # da = xr.tutorial.open_dataset("air_temperature")
    dataset = generate_temperature_data()
    dataset = dataset.air

    def diff(y):
        return y.flatten().sum()

    output = (
        dataset.to_series()
        .wax.stream(return_state=False)
        .apply(diff, format_dims=dataset.dims)
    )
    check = dataset.to_series().sum(level=0)
    assert_tree_all_close(output, check)


def test_ewm_dataframe():
    register_wax_accessors()
    T = int(1.0e2)
    N = 10
    dataframe = pd.DataFrame(
        onp.random.normal(size=(T, N)), index=pd.date_range("1970", periods=T)
    )
    y = dataframe.ewm(alpha=1.0 / 10.0).mean()
    y2 = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
    assert (y - y2).abs().stack().max() < 1.0e-6


def test_ewm_dataframe_no_format_outputs():
    register_wax_accessors()

    T = int(1.0e3)
    N = 10
    dataframe = pd.DataFrame(
        onp.random.normal(size=(T, N)), index=pd.date_range("1970", periods=T)
    )
    y = dataframe.ewm(alpha=1.0 / 10.0).mean()
    y2 = dataframe.wax.ewm(alpha=1.0 / 10.0, format_outputs=False).mean()
    y2 = pd.DataFrame(onp.array(y2), index=y.index)

    assert (y - y2).abs().stack().max() < 1.0e-6

    y2 = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()

    from wax.format import format_dataframe, format_series

    dataset = xr.DataArray(y2)
    output = y2.values

    # format dataframe
    # with no format_dims
    output_dataframe = format_dataframe(dataset.coords, output)
    assert isinstance(output_dataframe, pd.DataFrame)
    assert (output_dataframe.values == output).flatten().all()

    # with empty coords
    output_dataframe = format_dataframe(None, output)
    assert isinstance(output_dataframe, pd.DataFrame)
    assert (output_dataframe.values == output).flatten().all()

    # with explicit format_dims, but with nested data structure.
    # test when a nested data structure is passed to format_dataframe.
    outputs = {"output": output}
    format_dims = {"output": onp.array(dataset.dims)}
    outputs_dataframe = format_dataframe(dataset.coords, outputs, format_dims)
    assert isinstance(outputs_dataframe["output"], pd.DataFrame)
    assert (outputs_dataframe["output"].values == outputs["output"]).flatten().all()

    # format series
    # with explicit format_dims, but with nested data structure.
    # test when a nested data structure is passed to format_dataframe.
    outputs = {"output": output.sum(1)}
    format_dims = {"output": onp.array(dataset.dims)[0]}
    outputs_dataframe = format_series(dataset.coords, outputs, format_dims)
    assert isinstance(outputs_dataframe["output"], pd.Series)
    assert (outputs_dataframe["output"].values == outputs["output"]).flatten().all()
