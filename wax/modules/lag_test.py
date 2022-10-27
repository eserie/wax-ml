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
import haiku as hk
import jax
import pytest
from jax import numpy as jnp

from wax.compile import jit_init_apply
from wax.modules.lag import Lag
from wax.unroll import unroll


@pytest.mark.parametrize("use_jit", [False, True])
def test_lag(use_jit):
    seq = hk.PRNGSequence(42)
    x = jax.random.normal(next(seq), (2, 3))

    @hk.transform_with_state
    def lag(x):
        return Lag(1)(x)

    if use_jit:
        lag = jit_init_apply(lag)
    params, state = lag.init(next(seq), x)
    output, state = lag.apply(params, state, next(seq), x)
    assert len(output) == 2
    assert jnp.isnan(output).all()

    x1 = x
    x = jax.random.normal(next(seq), (2, 3))
    output, state = lag.apply(params, state, next(seq), x)
    assert len(output) == 2
    assert (output == (x1)).all()


def test_lag_unroll_int():
    xs = jnp.array([10, 11, 12], dtype="int32")

    res = unroll(lambda x: Lag(1)(x))(xs)
    assert (res == jnp.array([0, 10, 11])).all()

    res = unroll(lambda x: Lag(2)(x))(xs)
    assert (res == jnp.array([0, 0, 10])).all()


def test_lag_unroll_float():
    xs = jnp.array([10, 11, 12], dtype="float32")

    res = unroll(lambda x: Lag(1)(x))(xs)
    assert jnp.allclose(res, jnp.array([jnp.nan, 10, 11]), equal_nan=True)

    res = unroll(lambda x: Lag(2)(x))(xs)
    assert jnp.allclose(res, jnp.array([jnp.nan, jnp.nan, 10]), equal_nan=True)

    res = unroll(lambda x: Lag(1)(Lag(1)(x)))(xs)
    assert jnp.allclose(res, jnp.array([jnp.nan, jnp.nan, 10]), equal_nan=True)
