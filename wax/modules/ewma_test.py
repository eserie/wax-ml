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
import jax.numpy as jnp
import pandas as pd
import pytest
from jax.config import config

from wax.compile import jit_init_apply
from wax.modules.ewma import EWMA
from wax.unroll import dynamic_unroll_fori_loop, unroll


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_init_and_first_step_ema(dtype):

    if dtype == "float64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(3,), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust=True)(x)

    params, state = model.init(next(seq), x)
    ema, state = model.apply(params, state, next(seq), x)
    assert ema.dtype == jnp.dtype(dtype)


def test_run_ema_vs_pandas_not_adjust():

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust=False)(x)

    ema, state = unroll(model, dynamic=False, return_final_state=True)(x)

    pandas_ema = pd.DataFrame(x).ewm(alpha=0.1, adjust=False).mean()

    assert jnp.allclose(ema, pandas_ema.values)


def test_dynamic_unroll_fori_loop():

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust=True)(x)

    ema, state = unroll(model, dynamic=False, return_final_state=True)(x)

    ema2, state2 = dynamic_unroll_fori_loop(model, None, None, next(seq), False, x)

    assert jnp.allclose(ema, ema2)


def test_dynamic_unroll():

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust=True)(x)

    ema, state = unroll(model, dynamic=False, return_final_state=True)(x)

    seq = hk.PRNGSequence(42)
    ema2, state2 = unroll(model, return_final_state=True)(x)

    assert jnp.allclose(ema, ema2)


def test_run_ema_vs_pandas_adjust():

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust=True)(x)

    ema, state = unroll(model, return_final_state=True)(x)

    pandas_ema = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).mean()
    assert jnp.allclose(ema, pandas_ema.values)


def test_run_ema_vs_pandas_adjust_finite():

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMA(0.1, adjust="linear")(x)

    ema, state = unroll(model, return_final_state=True)(x)
    pandas_ema_adjust = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).mean()
    pandas_ema_not_adjust = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).mean()
    assert not jnp.allclose(ema, pandas_ema_adjust.values)
    assert not jnp.allclose(ema, pandas_ema_not_adjust.values)
    corr = jnp.corrcoef(ema.flatten(), pandas_ema_adjust.values.flatten())[0, 1]
    assert 1.0e-3 < 1 - corr < 1.0e-2


@pytest.mark.parametrize("adjust", [False, True, "linear"])
def test_grad_ewma(adjust):
    from functools import partial

    import jax
    import jax.numpy as jnp

    from wax.unroll import unroll_transform_with_state

    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (10, 3))
    _, rng = jax.random.split(rng)
    params = {"w": jax.random.normal(rng, (10,))}

    # put some nan values
    x = jax.ops.index_update(x, 0, jnp.nan)

    @partial(unroll_transform_with_state, dynamic=False)
    def fun(x):
        return EWMA(1 / 10, adjust=adjust)(x)

    # print("init")
    params, state = fun.init(rng, x)

    # print("apply")
    res, final_state = fun.apply(params, state, rng, x)
    # print(res)

    # print("gradient")
    @jax.value_and_grad
    def batch(params):
        res, final_state = fun.apply(params, state, rng, x)
        return res.mean()

    score, grad = batch(params)
    assert not jnp.isnan(grad["ewma"]["alpha"])
