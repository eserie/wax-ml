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
from functools import partial

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import pytest
from jax.config import config

from wax.compile import jit_init_apply
from wax.modules.ewma import EWMA
from wax.unroll import dynamic_unroll_fori_loop, unroll, unroll_transform_with_state


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
        return EWMA(alpha=0.1, adjust=True)(x)

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
        return EWMA(alpha=0.1, adjust=False)(x)

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
        return EWMA(alpha=0.1, adjust=True)(x)

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
        return EWMA(alpha=0.1, adjust=True)(x)

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
        return EWMA(alpha=0.1, adjust=True)(x)

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
        return EWMA(alpha=0.1, adjust="linear")(x)

    ema, state = unroll(model, return_final_state=True)(x)
    pandas_ema_adjust = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).mean()
    pandas_ema_not_adjust = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).mean()
    assert not jnp.allclose(ema, pandas_ema_adjust.values)
    assert not jnp.allclose(ema, pandas_ema_not_adjust.values)
    corr = jnp.corrcoef(ema.flatten(), pandas_ema_adjust.values.flatten())[0, 1]
    assert 1.0e-3 < 1 - corr < 1.0e-2


@pytest.mark.parametrize("adjust", [False, True, "linear"])
def test_grad_ewma(adjust):
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (10, 3))
    # put some nan values
    x = x.at[0].set(jnp.nan)

    _, rng = jax.random.split(rng)

    @partial(unroll_transform_with_state, dynamic=False)
    def fun(x):
        return EWMA(alpha=1 / 10, adjust=adjust)(x)

    params, state = fun.init(rng, x)
    res, final_state = fun.apply(params, state, rng, x)

    @jax.value_and_grad
    def batch(params):
        res, final_state = fun.apply(params, state, rng, x)
        return res.mean()

    score, grad = batch(params)
    assert not jnp.isnan(grad["ewma"]["logcom"])


def check_against_pandas_ewm(x, **ewma_kwargs):
    @partial(unroll_transform_with_state, dynamic=True)
    def fun(x):
        return EWMA(return_info=True, **ewma_kwargs)(x)

    rng = jax.random.PRNGKey(42)
    params, state = fun.init(rng, x)
    (res, info), final_state = fun.apply(params, state, rng, x)
    res = pd.DataFrame(onp.array(res))

    ref_res = pd.DataFrame(onp.array(x)).ewm(**ewma_kwargs).mean()
    pd.testing.assert_frame_equal(res, ref_res, atol=1.0e-6)

    @jax.value_and_grad
    def batch(params):
        (res, info), final_state = fun.apply(params, state, rng, x)
        return jnp.nanmean(res)

    score, grad = batch(params)
    assert not jnp.isnan(grad["ewma"]["logcom"])


@pytest.mark.parametrize(
    "adjust, ignore_na",
    [(False, False), (False, True), (True, False), (True, True)],  # ,
)
def test_nan_at_beginning(adjust, ignore_na):
    config.update("jax_enable_x64", True)

    T = 20
    x = jnp.full((T,), jnp.nan).at[2].set(1).at[10].set(-1)
    check_against_pandas_ewm(x, com=10, adjust=adjust, ignore_na=ignore_na)

    # check min_periods option with random variable
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (5,))
    check_against_pandas_ewm(
        x,
        com=10,
        adjust=adjust,
        ignore_na=ignore_na,
        min_periods=2,
    )

    # check random variable with nans
    # rng = jax.random.PRNGKey(42)
    # x = jax.random.normal(rng, (6,)).at[3].set(jnp.nan)
    # x = jnp.ones((6,), "float64").at[0].set(-1).at[3].set(jnp.nan)
    x = jnp.ones((30,), "float64").at[0].set(-1).at[5:20].set(jnp.nan)
    check_against_pandas_ewm(
        x,
        com=10,
        adjust=adjust,
        ignore_na=ignore_na,
    )


def test_init_value():
    x = (
        jnp.ones((30,), "float64")
        .at[0]
        .set(jnp.nan)
        .at[1]
        .set(-1)
        .at[5:20]
        .set(jnp.nan)
    )

    res = unroll(lambda x: EWMA(com=10, adjust=False, ignore_na=False)(x))(x)
    res_init0 = unroll(
        lambda x: EWMA(com=10, adjust=False, ignore_na=False, initial_value=0.0)(x)
    )(x)

    assert res_init0[0] == 0
    assert jnp.isnan(res[0])
    assert jnp.linalg.norm(res_init0) < jnp.linalg.norm(jnp.nan_to_num(res))


def test_train_ewma():
    import optax
    from tqdm.auto import tqdm

    COM_INIT = 100
    COM_TARGET = 10
    T = 1000
    n_epochs = 100

    def train():
        def model(x):
            return EWMA(com=COM_INIT, adjust=True, ignore_na=False)(x)

        @jax.jit
        def loss(y, y_ref):
            return jnp.nanmean((y - y_ref) ** 2)

        @jax.jit
        def loss_p(params, state, x, y_ref):
            y_pred, state = model.apply(params, state, rng, x)
            return loss(y_pred, y_ref)

        rng = jax.random.PRNGKey(42)
        x = jax.random.normal(rng, (T,)).at[T // 2].set(jnp.nan)

        y_ref = unroll(lambda x: EWMA(com=COM_TARGET, adjust=True, ignore_na=False)(x))(
            x
        )

        model = unroll_transform_with_state(model)
        params, state = model.init(rng, x)

        y_pred, _ = model.apply(params, state, rng, x)

        opt = optax.adagrad(1.0e-1)
        opt_state = opt.init(params)
        losses = []
        for e in tqdm(range(n_epochs)):
            # params, state = model.init(rng, x)
            _, state = model.init(rng, x)
            y_pred, state = model.apply(params, state, rng, x)
            l_ = loss(y_pred, y_ref)
            grad = jax.grad(loss_p)(params, state, x, y_ref)
            logcom = params["ewma"]["logcom"]
            if e % 100 == 0:
                print(
                    f"e={e}, logcom={logcom}, com={jnp.exp(logcom)}, grad={grad['ewma']['logcom']}, loss = {l_}"
                )
            updates, opt_state = opt.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            losses.append(l_)
        return jnp.array(losses)

    losses = train()
    assert losses[-1] < losses[0]
