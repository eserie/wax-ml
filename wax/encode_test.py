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
import datetime

import jax.numpy as jnp
import numpy as onp
import pandas as pd
import pytest

from wax.encode import (
    decode_datetime,
    decode_datetime64,
    decode_int64,
    decode_timestamp,
    encode_datetime,
    encode_datetime64,
    encode_int64,
    encode_timestamp,
    floor_datetime,
    floor_jax_datekey,
    less,
)


@pytest.mark.parametrize("seed", [10.0])
def test_encode_float(seed):
    with pytest.raises(TypeError):
        encode_int64(seed)


@pytest.mark.parametrize(
    "seed", [2 ** 63 - 1, 2 ** 50 - 46, 2 ** 34, onp.int32(10), onp.int64(10), 10]
)
def test_encode(seed):
    i32 = encode_int64(seed)
    assert i32 is not seed
    assert i32.dtype == onp.uint32
    assert i32.shape == (2,)
    i64 = decode_int64(i32)
    assert i64 == seed


@pytest.mark.parametrize("seed", [onp.int32(10), onp.int64(10), 10])
def test_encode_int32(seed):
    i32 = encode_int64(seed)
    assert i32 is not seed
    assert i32.dtype == onp.uint32
    assert i32.shape == (2,)
    high, low = i32
    assert high == 0
    assert low == 10
    i64 = decode_int64(i32)
    assert i64 == seed


def test_encode_timestamp():
    date = pd.to_datetime(datetime.datetime.now())
    code = encode_timestamp(date)
    decoded_date = decode_timestamp(code)
    assert type(decoded_date) == type(date)
    assert decoded_date == date


def test_encode_datetime():
    date = datetime.datetime.now()
    code = encode_datetime(date)
    decoded_date = decode_datetime(code)
    assert type(decoded_date) == type(date)
    assert decoded_date == date


@pytest.mark.parametrize("dtype", ["<M8[us]", "<M8[ns]"])
def test_encode_datetime64_simple(dtype):
    date = onp.datetime64(datetime.datetime.now()).astype(dtype)
    code = encode_datetime64(date)
    decoded_date = decode_datetime64(code)
    assert type(decoded_date) == type(date)
    assert decoded_date == date


TEST_INT_LIST = [
    (1, 2),
    (10, 2 ** 33),
    (2 ** 32, 2 ** 32 + 1),
    (2 ** 32 - 1, 2 ** 32),
]

# inverse order of t1, t2
TEST_INT_LIST += [(t2, t1) for t1, t2 in TEST_INT_LIST]


@pytest.mark.parametrize("t1, t2", TEST_INT_LIST)
def test_less(t1, t2):
    t1_is_less_t2 = t1 < t2
    ct1, ct2 = map(encode_int64, (t1, t2))
    assert less(ct1, ct2) == t1_is_less_t2

    # also check encode / decode
    assert decode_int64(ct1) == t1
    assert decode_int64(ct2) == t2


@pytest.mark.parametrize("dtype", ["<M8[us]", "<M8[ns]"])
def test_encode_datetime64(dtype):
    date = onp.datetime64(datetime.datetime.now()).astype(dtype)
    code = encode_datetime64(date)
    code = jnp.array(code)  # convert to jax DeviceArray
    assert isinstance(code, jnp.DeviceArray)

    # check decode
    decoded_date = decode_datetime64(code)
    # we are returned to numpy datetime
    assert type(decoded_date) == type(date)
    assert decoded_date == date

    # now check projection to date
    code_proj = floor_jax_datekey(code)
    date_proj = decode_datetime64(onp.array(code_proj))
    assert date_proj == floor_datetime(date)


def test_encode_datetime64_raises():
    with pytest.raises(TypeError):
        encode_datetime64(onp.array(2.0))

    with pytest.raises(TypeError):
        encode_datetime(onp.array(2.0))

    with pytest.raises(TypeError):
        encode_int64(onp.array(2.0))

    with pytest.raises(TypeError):
        encode_int64(onp.eye(2))

    with pytest.raises(TypeError):
        encode_timestamp(onp.eye(2))
