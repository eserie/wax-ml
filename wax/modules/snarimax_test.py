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
from typing import Any, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
import pytest
from optax._src.base import OptState

from wax.modules import ARMA, SNARIMAX, GymFeedback, OnlineOptimizer, UpdateParams, VMap
from wax.modules.lag import tree_lag
from wax.modules.vmap import add_batch
from wax.optim.newton import newton
from wax.unroll import unroll_transform_with_state


def generate_arma():
    alpha = jnp.array([0.6, -0.5, 0.4, -0.4, 0.3])
    beta = jnp.array([0.3, -0.2])

    rng = jax.random.PRNGKey(42)
    eps = jax.random.normal(rng, (100,))
    sim = unroll_transform_with_state(lambda eps: ARMA(alpha, beta)(eps))
    params, state = sim.init(rng, eps)
    y, state = sim.apply(params, state, rng, eps)
    return y


def test_snarimax_not_implemented():
    sim = hk.transform_with_state(lambda y: SNARIMAX(1, 1, 0)(y))
    rng = jax.random.PRNGKey(42)

    y = jax.random.normal(rng, (1,))
    with pytest.raises(NotImplementedError):
        sim.init(rng, y)


def test_snarimax():
    y = generate_arma()

    def predict(y, X=None):
        return SNARIMAX(10, 0, 0)(y, X)

    sim = unroll_transform_with_state(predict)
    rng = jax.random.PRNGKey(42)
    params, state = sim.init(rng, y)
    (y_pred, _), state = sim.apply(params, state, rng, y)

    assert y_pred.shape == y.shape


def build_env():
    def env(action, obs):
        y_pred, eps = action, obs
        ar_coefs = jnp.array([0.6, -0.5, 0.4, -0.4, 0.3])
        ma_coefs = jnp.array([0.3, -0.2])

        y = ARMA(ar_coefs, ma_coefs)(eps)
        # prediction used on a fresh y observation.
        rw = -((y - y_pred) ** 2)

        env_info = {"y": y, "y_pred": y_pred}
        obs = y
        return rw, obs, env_info

    return env


def build_agent(time_series_model=None, opt=None):
    if time_series_model is None:

        def time_series_model(y, X):
            return SNARIMAX(10)(y, X)

    if opt is None:
        opt = newton(0.3, 0.3)

    class AgentInfo(NamedTuple):
        optim: Any
        forecast: Any

    class ModelWithLossInfo(NamedTuple):
        pred: Any
        loss: Any

    def agent(obs):
        if isinstance(obs, tuple):
            y, X = obs
        else:
            y = obs
            X = None

        def evaluate(y_pred, y):
            return jnp.linalg.norm(y_pred - y) ** 2, {}

        def model_with_loss(y, X=None):
            # predict with lagged data
            y_pred, pred_info = time_series_model(*tree_lag(1)(y, X))

            # evaluate loss with actual data
            loss, loss_info = evaluate(y_pred, y)

            return loss, ModelWithLossInfo(pred_info, loss_info)

        def project_params(params: Any, opt_state: OptState = None):
            del opt_state
            return jax.tree_util.tree_map(lambda w: jnp.clip(w, -1, 1), params)

        def params_predicate(m: str, n: str, p: jnp.ndarray) -> bool:
            # print(m, n, p)
            return m.endswith("snarimax/~/linear") and n == "w"

        def learn_and_forecast(y, X=None):
            opt_info = OnlineOptimizer(
                model_with_loss,
                opt,
                project_params=project_params,
                params_predicate=params_predicate,
                return_params=True,
            )(*tree_lag(1)(y, X))

            predict_params = opt_info.params

            y_pred, forecast_info = UpdateParams(time_series_model)(
                predict_params, y, X
            )
            return y_pred, AgentInfo(opt_info, forecast_info)

        return learn_and_forecast(y, X)

    return agent


def run_scan_hyper_params(env):
    n_time_step = 100
    n_batches = 3
    n_step_size = 2
    n_eps = 2

    STEP_SIZE = pd.Index(onp.logspace(-2, 3, n_step_size), name="step_size")
    EPS = pd.Index(onp.logspace(-4, 3, n_eps), name="eps")

    HPARAMS_idx = pd.MultiIndex.from_product([STEP_SIZE, EPS])
    HPARAMS = jnp.stack(list(map(onp.array, HPARAMS_idx)))

    @partial(add_batch, take_mean=False)
    def gym_loop_scan_hparams(eps):
        def scan_params(hparams):
            step_size, newton_eps = hparams
            agent = build_agent(opt=newton(step_size, eps=newton_eps))
            return GymFeedback(agent, env)(eps)

        return VMap(scan_params)(HPARAMS)

    sim = unroll_transform_with_state(gym_loop_scan_hparams)
    rng = jax.random.PRNGKey(42)
    eps = jax.random.normal(rng, (n_time_step, n_batches)) * 0.3

    params, state = sim.init(rng, eps)
    (gym, info), state = sim.apply(params, state, rng, eps)

    assert gym.reward.shape == (n_time_step, n_batches, n_step_size * n_eps)


def test_gym_loop():
    env = build_env()
    agent = build_agent()

    rng = jax.random.PRNGKey(42)
    eps = jax.random.normal(rng, (100,)) * 0.3

    sim = unroll_transform_with_state(lambda eps: GymFeedback(agent, env)(eps))
    params, state = sim.init(rng, eps)
    (gym, info), state = sim.apply(params, state, rng, eps)

    # pd.Series(-gym.reward).expanding().mean().plot(label="AR(10) model")
    assert -gym.reward.sum() > 0


def test_scan_hyper_params():
    env = build_env()
    run_scan_hyper_params(env)
