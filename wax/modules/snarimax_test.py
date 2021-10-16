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
from typing import Any, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import pandas as pd
from optax._src.base import OptState

from wax.modules import (
    ARMA,
    SNARIMAX,
    GymFeedback,
    Lag,
    OnlineOptimizer,
    UpdateParams,
    VMap,
)
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


def test_snarimax():
    y = generate_arma()

    def predict(y, X=None):
        return SNARIMAX(10, 0, 0)(y, X)

    sim = unroll_transform_with_state(predict)
    rng = jax.random.PRNGKey(42)
    params, state = sim.init(rng, y)
    (y_pred, _), state = sim.apply(params, state, rng, y)

    assert y_pred.shape == y.shape


def build_env_agent():
    class Env(hk.Module):
        def __call__(self, y_pred, eps):
            ar_coefs = jnp.array([0.6, -0.5, 0.4, -0.4, 0.3])
            ma_coefs = jnp.array([0.3, -0.2])

            y = ARMA(ar_coefs, ma_coefs)(eps)
            # prediction used on a fresh y observation.
            rw = -((y - y_pred) ** 2)
            return rw, y, {"y": y, "y_pred": y_pred}

    class ForecastInfo(NamedTuple):
        optim: Any
        forecast: Any

    def parametrized_learn_and_forecast(ts_model, opt):
        def learn_and_forecast(y, X=None):
            def predict(y, X=None):
                return ts_model(y, X)

            def evaluate(y_pred, y):
                return jnp.linalg.norm(y_pred - y) ** 2, {}

            def lag(shift=1):
                def __call__(y, X=None):
                    yp = Lag(shift)(y)
                    Xp = Lag(shift)(X) if X is not None else None
                    return yp, Xp

                return __call__

            def model_with_loss(y, X=None):
                # predict with lagged data
                y_pred, pred_info = predict(*lag(1)(y, X))

                # evaluate loss with actual data
                loss, loss_info = evaluate(y_pred, y)

                return loss, dict(pred_info=pred_info, loss_info=loss_info)

            def project_params(params: Any, opt_state: OptState = None):
                w = params["snarimax/~/linear"]["w"]
                w = jnp.clip(w, -1, 1)
                params["snarimax/~/linear"]["w"] = w
                return params

            def split_params(params):
                def filter_params(m, n, p):
                    # print(m, n, p)
                    return m == "snarimax/~/linear" and n == "w"

                return hk.data_structures.partition(filter_params, params)

            optim_res = OnlineOptimizer(
                model_with_loss,
                opt,
                project_params=project_params,
                split_params=split_params,
            )(*lag(1)(y, X))

            predict_params = optim_res.updated_params

            forecast, forecast_info = UpdateParams(predict)(predict_params, y, X)
            return forecast, ForecastInfo(optim_res, forecast_info)

        return learn_and_forecast

    class Agent(hk.Module):
        def __init__(self, ts_model, opt=None, name=None):
            super().__init__(name=name)
            self.ts_model = ts_model
            self.opt = opt if opt is not None else optax.sgd(1.0e-3)

        def __call__(self, y):
            y_pred, info = parametrized_learn_and_forecast(self.ts_model, self.opt)(y)
            return y_pred, info

    return Env, Agent


def test_gym_lopp():
    Env, Agent = build_env_agent()

    def gym_loop(eps):
        return GymFeedback(Agent(SNARIMAX(10, 0, 0), newton(0.3, 0.3)), Env())(eps)

    rng = jax.random.PRNGKey(42)
    eps = jax.random.normal(rng, (100,)) * 0.3
    sim = unroll_transform_with_state(gym_loop)
    params, state = sim.init(rng, eps)
    (gym, info), state = sim.apply(params, state, rng, eps)
    # pd.Series(-gym.reward).expanding().mean().plot(label="AR(10) model")
    assert -gym.reward.sum() > 0


def test_scan_hyper_params():
    Env, Agent = build_env_agent()

    n_time_step = 100
    n_batches = 3
    n_step_size = 2
    n_eps = 2

    STEP_SIZE = pd.Index(onp.logspace(-2, 3, n_step_size), name="step_size")
    EPS = pd.Index(onp.logspace(-4, 3, n_eps), name="eps")

    HPARAMS_idx = pd.MultiIndex.from_product([STEP_SIZE, EPS])
    HPARAMS = jnp.stack(list(map(onp.array, HPARAMS_idx)))

    def gym_loop_scan_hyper_param(eps):
        def batch_eps(eps):
            def scan_params(hparams):
                step_size, newton_eps = hparams
                opt = newton(step_size, eps=newton_eps)
                return GymFeedback(Agent(SNARIMAX(10, 0, 0), opt=opt), Env())(eps)

            res = VMap(scan_params)(HPARAMS)
            return res

        # uncomment to take the mean
        # return jax.tree_map(lambda x: x.mean(axis=0), VMap(batch_eps)(eps))
        return VMap(batch_eps)(eps)

    sim = unroll_transform_with_state(gym_loop_scan_hyper_param)
    rng = jax.random.PRNGKey(42)
    eps = jax.random.normal(rng, (n_time_step, n_batches)) * 0.3

    params, state = sim.init(rng, eps)
    (gym, info), state = sim.apply(params, state, rng, eps)

    assert gym.reward.shape == (n_time_step, n_batches, n_step_size * n_eps)
