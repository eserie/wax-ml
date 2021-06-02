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
from functools import partial, reduce

import haiku as hk
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import xarray as xr
from haiku import transform_with_state
from jax import tree_flatten
from jax.numpy import array as DeviceArray
from jax.numpy import int32
from jax.tree_util import tree_map
from numpy import array
from pandas import NaT

import wax.external.eagerpy as ep
from wax.datasets.generate_temperature_data import (
    generate_temperature_data_multi_time_scale,
)
from wax.encode import encode_dataset
from wax.stream import (
    Stream,
    dataset_to_numpy,
    get_time_dataset,
    split_dataset_from_time_dims,
    tree_access_data,
    unroll_stream,
)
from wax.testing import assert_tree_all_close
from wax.unroll import dynamic_unroll
from wax.utils import dict_map


def onp_half_precision(x):
    if x.dtype == onp.float64:
        return x.astype(onp.float32)
    if x.dtype == onp.int64:
        return x.astype(onp.int32)
    return x


def test_onp_half_precision():
    x = onp.array(1, dtype=onp.int32)
    assert onp_half_precision(x) == x


def generate_news_data():
    days = 2
    bins = 1000
    onp.random.seed(3)
    dates = pd.date_range(start="2000/01/02", periods=days, freq="B")
    news_times = pd.Index(
        reduce(
            lambda x, y: x + y,
            [
                (date + pd.timedelta_range(start=1, freq="1min", periods=bins)).tolist()
                for date in dates
            ],
        )
    )
    idx_news = onp.random.choice(len(news_times), 10)
    news_times = news_times[sorted(idx_news)]

    # create a bin with 5 observations
    news_times = news_times.tolist()
    news_times.append(news_times[0] + onp.timedelta64(3, "s"))
    news_times.append(news_times[0] + onp.timedelta64(10, "s"))
    news_times.append(news_times[0] + onp.timedelta64(20, "s"))
    news_times.append(news_times[0] + onp.timedelta64(27, "s"))
    news_times = pd.Index(sorted(news_times)).values
    news_times = news_times + onp.timedelta64(50, "ms")
    T = len(news_times)
    news_data = onp.random.normal(size=(T, 2)).astype(str)
    news_data = xr.DataArray(
        news_data,
        dims=("news_time", "news_attr"),
        coords={"news_time": news_times, "news_attr": ["head", "text"]},
    )
    return news_data


def prepare_test_data():
    xdata = generate_temperature_data_multi_time_scale()

    xdata["NEWS"] = generate_news_data()

    xdata["param"] = 1

    return xdata


def check_time_index(index, direct=True):
    if direct:
        name_news = "news_time_index"
        name_air = "day_index"
    else:
        name_news = "NEWS"
        name_air = "air"
    idx = index["news_time"].value[name_news]
    assert len(onp.where(idx > 0)[0].flatten()) == 27

    ref = index["news_time"].time.tolist()
    assert index["news_time"].time.shape == (124, 3)

    assert ref[0] == [NaT, NaT, NaT]
    assert ref[53] == [946872970050000001, 946872980050000001, 946872987050000001]

    ref = index["news_time"].value[name_news].tolist()
    assert ref[0] == [-99999, -99999, -99999]
    assert ref[53] == [2, 3, 4]

    assert index["news_time"].value[name_news].shape == (124, 3)

    ref = index["day"].time.tolist()[20:30]
    assert ref == [
        NaT,
        NaT,
        NaT,
        NaT,
        946684800000000000,
        946684800000000000,
        946684800000000000,
        946684800000000000,
        946684800000000000,
        946684800000000000,
    ]

    ref = index["day"].value[name_air].tolist()[20:30]
    assert (ref == array([-99999, -99999, -99999, -99999, 0, 0, 0, 0, 0, 0])).all()


def check_outputs(outputs, schema):

    # TODO: check what is correct...
    # assert outputs["NEWS"].shape == (25, 3, 2)
    assert outputs["NEWS"].shape == (124, 3)
    assert_tree_all_close(
        outputs["NEWS"][-1],
        DeviceArray([-99999, -99999, -99999], dtype=int32),
    )
    assert_tree_all_close(
        outputs["NEWS"][53],
        DeviceArray([27, 6, 23], dtype=int32),
    )

    assert_tree_all_close(
        outputs["NEWS"][:, 0][60:70],
        DeviceArray(
            [-99999, -99999, 23, -99999, 10, 1, -99999, -99999, -99999, -99999],
            dtype=int32,
        ),
    )

    # decode outputs
    assert outputs["NEWS"].shape == (124, 3)


def test_stream_dataset_dynamic_unroll2():
    dataset = prepare_test_data()
    assert dataset["NEWS"].shape == (14, 2)
    schema = Stream.get_dataset_schema(dataset, check=True)
    ffills = {"day": True}
    freqs = {"day": "d"}
    local_time = "time"
    buffer_maxlen = {"news_time": 3}

    datasets = split_dataset_from_time_dims(dataset)
    assert datasets.keys() == {"day", "news_time", "time"}
    assert datasets["news_time"]["NEWS"].shape == (14, 2)

    stream = Stream(
        local_time,
        freqs,
        ffills,
        buffer_maxlen,
        verbose=True,
        pbar=["time"],
        trace=True,
    )
    streams = {}
    for time_dim, ds in datasets.items():
        streams[time_dim] = stream.start_dataset_stream(ds, time_dim=time_dim)
    trace = stream.merge(streams)
    index = unroll_stream(trace)

    def check_type(data):
        data, treedef = tree_flatten(data)
        for x in data:
            assert isinstance(x, onp.ndarray)

    check_type(index)

    check_time_index(index, False)

    # convert to dict of numpy

    np_data = dict_map(dataset_to_numpy, datasets)

    # encode values
    assert datasets["news_time"]["NEWS"].shape == (14, 2)
    np_data = dict_map(partial(encode_dataset, schema.encoders), np_data)
    assert np_data["news_time"]["NEWS"].shape == (28,)
    # drop times from index structure
    np_index = {dim: var.value for dim, var in index.items()}

    # explicitly convert in np.float32 and int32 berfore jax conversion to avoid jax warnings.
    np_data, np_index = tree_map(onp_half_precision, (np_data, np_index))
    # convert to jax
    np_data, np_index = ep.convert_to_tensors((np_data, np_index), "jax")

    # check data types:
    def check_type(data):
        data, treedef = tree_flatten(data)
        for x in data:
            assert isinstance(x, jnp.DeviceArray)

    check_type(np_data)
    check_type(np_index)

    access_dataset = transform_with_state(partial(tree_access_data, np_data, np_index))

    xs = onp.arange(len(schema.coords["time"]))
    seq = hk.PRNGSequence(42)

    # outputs, state = static_unroll(access_dataset, xs, next(seq))
    # now test dynamic unroll
    outputs, state = dynamic_unroll(access_dataset, None, None, next(seq), False, xs)

    check_outputs(outputs["news_time"], schema)


def test_stream_dataset_dynamic_unroll3():
    dataset = prepare_test_data()
    assert dataset["NEWS"].shape == (14, 2)
    schema = Stream.get_dataset_schema(dataset, check=True)
    ffills = {"day": True}
    freqs = {"day": "d"}
    local_time = "time"
    buffer_maxlen = {"news_time": 3}

    time_dataset = get_time_dataset(dataset)

    time_datasets = split_dataset_from_time_dims(time_dataset)
    stream = Stream(
        local_time, freqs, ffills, buffer_maxlen, verbose=["time"], pbar=True
    )
    streams = stream.start_dataset_streams(time_datasets)
    trace = stream.merge(streams)
    time_index = unroll_stream(trace)

    # check time_index
    def check_type(data):
        data, treedef = tree_flatten(data)
        for x in data:
            assert isinstance(x, onp.ndarray)

    check_type(time_index)
    check_time_index(time_index)

    from .stream import get_dataset_index, get_dataset_index_from_stream_index

    # now convert in index for the original dataset.
    time_dataset_index = get_dataset_index_from_stream_index(time_index)
    dataset_index = get_dataset_index(dataset, time_dataset_index)
    assert dataset["param"].shape == ()

    assert dataset_index["param"].shape == (124,)
    # check that dataset_index and dataset have same variables.
    assert dataset.keys() == dataset_index.keys()

    # convert to dict of numpy

    np_data = dataset_to_numpy(dataset)
    np_index = dataset_to_numpy(dataset_index)
    assert np_index["param"].shape == (124,)

    # encode values
    assert dataset["NEWS"].shape == (14, 2)
    np_data_encoded = encode_dataset(schema.encoders, np_data)
    from wax.encode import decode_dataset

    np_data_reencoded = decode_dataset(schema.encoders, np_data_encoded)
    for key in np_data:
        assert (np_data[key] == np_data_reencoded[key]).all()
    np_data = np_data_encoded

    assert np_data["NEWS"].shape == (28,)
    # drop times from index structure

    # explicitly convert in np.float32 and int32 berfore jax conversion to avoid jax warnings.
    np_data, np_index = tree_map(onp_half_precision, (np_data, np_index))
    # convert to jax
    np_data, np_index = ep.convert_to_tensors((np_data, np_index), "jax")

    # check data types:
    def check_type(data):
        data, treedef = tree_flatten(data)
        for x in data:
            assert isinstance(x, jnp.DeviceArray)

    check_type(np_data)
    check_type(np_index)
    assert np_index["param"].shape == (124,)
    access_dataset = transform_with_state(partial(tree_access_data, np_data, np_index))

    xs = onp.arange(len(schema.coords["time"]))
    seq = hk.PRNGSequence(42)

    # outputs, state = static_unroll(access_dataset, xs, next(seq))
    # now test dynamic unroll
    outputs, state = dynamic_unroll(access_dataset, None, None, next(seq), False, xs)

    # decode outputs
    assert outputs["NEWS"].shape == (124, 3)
    check_outputs(outputs, schema)
