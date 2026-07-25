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
"""Encoding schemes to encode/decode numpy data types non supported by JAX, e.g.
`datetime64` and `string_` dtypes."""

import datetime
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

import numpy as onp
import pandas as pd
from jax import numpy as jnp
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import LabelEncoder

DTypeLike = TypeVar("DTypeLike")


def encode_int64(seed: int) -> onp.ndarray:
    """Encode an int64 into in a 2-dimensional int32 ndarray.

    Args:
      seed: a 64- or 32-bit integer.

    Returns:
      an array of shape (2,) and dtype uint32.

    References
        See
        - jax implementation of PRNGKey.
        https://jax.readthedocs.io/en/latest/_autosummary/jax.random.PRNGKey.html  # noqa
        - https://codereview.stackexchange.com/questions/80386/packing-and-unpacking-two-32-bit-integers-into-an-unsigned-64-bit-integer  # noqa
    Note:
        0xFFFFFFFF = 2**32 -1
    """
    if onp.shape(seed):
        raise TypeError("seed must be a scalar.")
    if isinstance(seed, onp.ndarray):
        seed = seed.item()
    if not isinstance(seed, int | onp.int32 | onp.uint32 | onp.int64):
        raise TypeError(f"seed must be an int, got {type(seed)}")

    def _convert(k):
        return onp.reshape(onp.array(k, dtype=onp.uint32), [1])

    # Convert to int64 first to handle large numbers properly
    seed_i64 = onp.int64(seed)

    # Extract high and low 32-bit parts
    high = _convert((seed_i64 >> 32) & 0xFFFFFFFF)
    low = _convert(seed_i64 & 0xFFFFFFFF)
    return onp.concatenate([high, low], 0)


def decode_int64(code: onp.ndarray | jnp.ndarray) -> int:
    """See https://codereview.stackexchange.com/questions/80386/packing-and-unpacking-two-32-bit-integers-into-an-unsigned-64-bit-integer  # noqa"""
    # assert isinstance(code, np.ndarray)
    code_arr = onp.array(code)
    high, low = code_arr[0], code_arr[1]

    # Convert to int64 and reconstruct the original 64-bit integer
    high_i64 = onp.int64(high) << 32
    low_i64 = onp.int64(low)
    result = high_i64 + low_i64

    return int(result)


def encode_timestamp(date):
    if not isinstance(date, pd.Timestamp):
        raise TypeError(f"seed must be a pandas Timestamp, got {type(date)}")
    return encode_int64(date.value)


def decode_timestamp(code):
    return pd.to_datetime(decode_int64(code))


def encode_datetime(date):
    if not isinstance(date, datetime.datetime):
        raise TypeError(f"seed must be a datetime, got {type(date)}")
    return encode_datetime64(onp.datetime64(date))


def decode_datetime(code):
    return pd.to_datetime(decode_int64(code)).to_pydatetime()


def encode_datetime64(date):
    # Check if it's a datetime64 type more robustly
    if not onp.issubdtype(date.dtype, onp.datetime64):
        raise TypeError(f"seed must be a np.datetime64, got {type(date)} with dtype {date.dtype}")
    return encode_int64(date.astype("<M8[ns]").astype(onp.int64))


def decode_datetime64(code):
    return onp.int64(decode_int64(code)).astype("<M8[ns]")


class Encoder(NamedTuple):
    encode: Callable
    decode: Callable


class DecodeDataset(NamedTuple):
    iter_coords: dict
    embed_map: dict
    event_map: dict
    embedding_coords: dict
    event_coords: dict
    other_coords: dict
    encoders: dict
    constant_data: dict
    iter_data: dict
    embed_data: dict
    event_data: dict


def floor_datetime(time, freq: str | DateOffset = "d"):
    r"""Perform floor operation on the data to the specified freq.
    Args:
        time : time(s) to floor.
        freq : The frequency level to floor the index to. Must be a fixed frequency
            like ‘S’ (second) not ‘ME’ (month end).
            See frequency aliases for a list of possible freq values.

    See:
    https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.floor.html

    As discussed in:
    https://stackoverflow.com/questions/5476065/how-to-truncate-the-time-on-a-datetime-object-in-python  # noqa
    the implementation with pd.Series.dt.florr seems to be the most performant.

        times = pd.Series(pd.date_range(start='1/1/2018 04:00:00', end='1/1/2018 22:00:00', freq='s'))
        %timeit floor_datetime(times)
        791 µs ± 4.19 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

    https://docs.oracle.com/cd/B19306_01/server.102/b14200/functions201.htm
    """
    assert freq == "d", f"other formats than '{freq}' not yet supported."
    # v3 (only in numpy)
    time_int64 = time.astype("<M8[ns]").astype(onp.int64)
    day = onp.array(24 * 60 * 60 * 1.0e9, onp.int64)
    floor_time_int64 = onp.array((time_int64 // day) * day, onp.int64)
    floor_time = floor_time_int64.astype("<M8[ns]")
    return floor_time

    # v1
    # assert freq == "d", f"other formats than '{freq}' not yet supported"
    # return pd.to_datetime(time).round(freq).to_numpy().astype("<M8[ns]")

    # v2
    # return pd.Series(time).dt.floor(freq).values.reshape(time.shape).astype("<M8[ns]")


def string_encoder(values: Any) -> Encoder:
    ravel = False
    original_shape = None

    if values.ndim > 1:
        ravel = True
        original_shape = values.shape
        values = values.ravel()
    encoder = LabelEncoder().fit(values)

    def encode(value):
        if ravel:
            value = value.ravel()
        return encoder.transform(value)

    def decode(code):
        value = encoder.inverse_transform(code)
        if ravel:
            value = value.reshape(original_shape)
        return value

    return Encoder(encode, decode)


def datetime64_encoder(values: Any) -> Encoder:
    def encode(value):
        return onp.stack(list(map(encode_datetime64, value)))

    def decode(code):
        return onp.stack(list(map(decode_datetime64, [code[i] for i in range(len(code))])))

    return Encoder(encode, decode)


def less(t1, t2):
    h1, l1 = t1
    h2, l2 = t2
    if h1 < h2:
        return True
    elif h1 > h2:
        return False
    else:
        if l1 < l2:
            return True
        else:
            return False


def floor_jax_datekey(datekey):
    """Floor a date represented as a datekey.

    TODO: find an efficient implementation which do not need
    to pass by int64 conversions in order to use it smoothly
    in Jax 32 bits worflows.

    """

    date = decode_int64(onp.array(datekey))
    day = onp.int64(24 * 60 * 60 * 1.0e9)
    floor_date = int((date // day) * day)
    datekey = encode_int64(floor_date)
    datekey = jnp.array(datekey)
    return datekey


def encode_dataset(encoders, dataset):
    output = {}
    for dim, var in dataset.items():
        if dim in encoders:
            # encode values
            values = encoders[dim].encode(var)
        else:
            values = var
        output[dim] = values
    return output


def decode_dataset(encoders, dataset):
    output = {}
    for dim, var in dataset.items():
        if dim in encoders:
            # encode values
            values = encoders[dim].decode(var)
        else:
            values = var
        output[dim] = values
    return output
