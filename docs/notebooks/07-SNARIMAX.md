---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->
# 🔄 Online learning for time series prediction 🔄

We implement the online learning filter developped in [1].

In the first part of this notebook, we reproduce
the "setting 1" of the paper and introduce the functions implemented in WAX-ML.
We finally reproduce the other settings (2, 3, 4) which permit to see how differnet algorithms behave in more challenging environments.


We use the following modules from WAX-ML:
- ARMA : to generate a modeled time-series
- SNARIMAX : to adaptively learn to predict the generated time-series.
- GymFeedback: To setup a training loop.
- VMap: to add batch dimensions to the training loop
- optim.newton: a newton algorithm as used in [1] and developped in [2]. It extends optax optimizers.

[1] [Anava, O., Hazan, E., Mannor, S. and Shamir, O., 2013, June. Online learning for time series prediction. In Conference on learning theory (pp. 172-184)](https://arxiv.org/pdf/1302.6927.pdf)

[2] [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for
online convex optimization. Machine Learning, 69(2-3), pp.169-192](https://link.springer.com/content/pdf/10.1007/s10994-007-5016-8.pdf)
<!-- #endregion -->

```python
%pylab inline
%load_ext autoreload
%autoreload 2
```


```python
from typing import Any, NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from wax.modules import (
    ARMA,
    SNARIMAX,
    GymFeedback,
    Lag,
    OnlineOptimizer,
    UpdateParams,
    VMap,
)
from wax.optim import newton
from wax.unroll import unroll_transform_with_state

```

# ARMA



Let's generate a sample of the "setting 1" of [1]:

```python
alpha = jnp.array([0.6, -0.5, 0.4, -0.4, 0.3])
beta = jnp.array([0.3, -0.2])

rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (1000,))
sim = unroll_transform_with_state(lambda eps: ARMA(alpha, beta)(eps))
params, state = sim.init(rng, eps)
y, state = sim.apply(params, state, rng, eps)
y.shape
```

# SNARIMAX



Let's setup an online model to try to learn the dynamic of the time-series.


First let's run the filter with it's initial random weights.

```python
def predict(y, X=None):
    return SNARIMAX(10, 0, 0)(y, X)


sim = unroll_transform_with_state(predict)
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (10,))
params, state = sim.init(rng, y)
(y_pred, _), state = sim.apply(params, state, rng, y)

pd.Series((y - y_pred)).plot()
```

```python
def evaluate(y_pred, y):
    return jnp.linalg.norm(y_pred - y) ** 2, {}


def lag(shift=1):
    def __call__(y, X=None):
        yp = Lag(shift)(y)
        Xp = Lag(shift)(X) if X is not None else None
        return yp, Xp

    return __call__


def predict_and_evaluate(y, X=None):
    # predict with lagged data
    y_pred, pred_info = predict(*lag(1)(y, X))

    # evaluate loss with actual data
    loss, loss_info = evaluate(y_pred, y)

    return loss, dict(pred_info=pred_info, loss_info=loss_info)
```

```python
sim = unroll_transform_with_state(predict_and_evaluate)
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (1000,))
params, state = sim.init(rng, y)
(loss, _), state = sim.apply(params, state, rng, y)

pd.Series(loss).expanding().mean().plot()
```

Since the model is not trained and the coefficient of the SNARIMAX filter a choosen randomly, the loss may diverge

```python
params
```

# Learn


## Setup projection

```python
def project_params(params, opt_state=None):
    w = params["snarimax/~/linear"]["w"]
    w = jnp.clip(w, -1, 1)
    params["snarimax/~/linear"]["w"] = w
    return params


project_params(hk.data_structures.to_mutable_dict(params))
```


```python
def learn(y, X=None):
    optim_res = OnlineOptimizer(
        predict_and_evaluate, optax.sgd(1.0e-2), project_params=project_params
    )(y, X)
    return optim_res
```

```python
sim = unroll_transform_with_state(learn)
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (1000,))
params, state = sim.init(rng, eps)
optim_res, state = sim.apply(params, state, rng, eps)

pd.Series(optim_res.loss).expanding().mean().plot()
```

Let's look at the latest weights:

```python
jax.tree_map(lambda x: x[-1], optim_res.updated_params)
```

# Learn and Forecast


```python
class ForecastInfo(NamedTuple):
    optim: Any
    forecast: Any
```

```python
def learn_and_forecast(y, X=None):
    optim_res = OnlineOptimizer(
        predict_and_evaluate, optax.sgd(1.0e-3), project_params=project_params
    )(*lag(1)(y, X))

    predict_params = optim_res.updated_params

    forecast, forecast_info = UpdateParams(predict)(predict_params, y, X)
    return forecast, ForecastInfo(optim_res, forecast_info)
```

```python
sim = unroll_transform_with_state(learn_and_forecast)
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (1000,))
params, state = sim.init(rng, y)
(forecast, info), state = sim.apply(params, state, rng, y)

pd.Series(info.optim.loss).expanding().mean().plot()
```

# Gym simulation


## Environment


let's build an environment corresponding to "setting 1" in [1]

```python


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
```

```python
env = build_env()
```


## Agent

```python
from optax._src.base import OptState
```

```python


def build_agent(time_series_model=None, opt=None):
    if time_series_model is None:
        time_series_model = lambda y, X: SNARIMAX(10)(y, X)

    if opt is None:
        opt = optax.sgd(1.0e-3)

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
            y_pred, pred_info = time_series_model(*lag(1)(y, X))

            # evaluate loss with actual data
            loss, loss_info = evaluate(y_pred, y)

            return loss, ModelWithLossInfo(pred_info, loss_info)

        def project_params(params: Any, opt_state: OptState = None):
            del opt_state
            return jax.tree_map(lambda w: jnp.clip(w, -1, 1), params)

        def split_params(params):
            def filter_params(m, n, p):
                # print(m, n, p)
                return m.endswith("snarimax/~/linear") and n == "w"

            return hk.data_structures.partition(filter_params, params)

        def learn_and_forecast(y, X=None):
            optim_res = OnlineOptimizer(
                model_with_loss,
                opt,
                project_params=project_params,
                split_params=split_params,
            )(*lag(1)(y, X))

            predict_params = optim_res.updated_params

            y_pred, forecast_info = UpdateParams(time_series_model)(
                predict_params, y, X
            )
            return y_pred, AgentInfo(optim_res, forecast_info)

        return learn_and_forecast(y, X)

    return agent
```


```python
agent = build_agent()
```


## Gym loop


```python
def gym_loop(eps):
    return GymFeedback(agent, env)(eps)
```

```python
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (10000,))
sim = unroll_transform_with_state(gym_loop)
params, state = sim.init(rng, eps)
(gym, info), state = sim.apply(params, state, rng, eps)
pd.Series(info.agent.optim.loss).expanding().mean().plot(label="agent loss")
pd.Series(-gym.reward).expanding().mean().plot(xlim=(0, 100), label="env loss")
plt.legend()
```

```python
pd.Series(-gym.reward).expanding().mean().plot()  # ylim=(0.09, 0.15))
```


We see that the agent suffers the same loss as the environment but with a time lag.


# Batch simulations


## Average over 20 experiments


## Slow version

First, let's do it "naively" by doing a simple python "for loop".

```python
%%time
rng = jax.random.PRNGKey(42)
sim = unroll_transform_with_state(gym_loop)

res = {}
for i in tqdm(onp.arange(20)):
    rng, _ = jax.random.split(rng)
    eps = jax.random.normal(rng, (10000,)) * 0.3
    params, state = sim.init(rng, eps)
    (gym_output, gym_info), final_state = sim.apply(params, state, rng, eps)
    res[i] = gym_info
```

```python
pd.DataFrame({k: pd.Series(v.agent.optim.loss) for k, v in res.items()}).mean(
    1
).expanding().mean().plot(ylim=(0.09, 0.15))
```

## Fast version with vmap

Instead of using a "for loop" we can use jax's vmap transformation function!


```python
%%time
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (20, 10000)) * 0.3

rng = jax.random.PRNGKey(42)
rng = jax.random.split(rng, num=20)
sim = unroll_transform_with_state(gym_loop)
params, state = jax.vmap(sim.init)(rng, eps)
(gym_output, gym_info), final_state = jax.vmap(sim.apply)(params, state, rng, eps)
```

This is much faster!

```python
pd.DataFrame(gym_info.agent.optim.loss).mean().expanding().mean().plot(
    ylim=(0.09, 0.15)
)
```

```python
i_batch = 0
w = gym_info.agent.optim.updated_params["snarimax/~/linear"]["w"][i_batch, :, :, 0]
w = pd.DataFrame(w)

ax = w.plot(title=f"weights on batch {i_batch}")
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel("time")
ax.set_ylabel("weight value")


plt.figure()
w.iloc[-1][::-1].plot(kind="bar")
```

```python
w = gym_info.agent.optim.updated_params["snarimax/~/linear"]["w"].mean(axis=0)[:, :, 0]
w = pd.DataFrame(w)
ax = w.plot(title="averaged weights (over batches)")
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel("time")
ax.set_ylabel("weight")

plt.figure()
w.iloc[-1][::-1].plot(kind="bar")
```

## With VMap module

We can use the wrapper module `VMap` of WAX-ML. It permits to have an ever simpler syntax.



> Note: we have to swap in position of time and batch dimension in the generation of eps.

```python
%%time
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (10000, 20)) * 0.3


def batched_gym_loop(eps):
    return VMap(gym_loop)(eps)


sim = unroll_transform_with_state(batched_gym_loop)

rng = jax.random.PRNGKey(43)
params, state = sim.init(rng, eps)
(gym_output, gym_info), final_state = sim.apply(params, state, rng, eps)
```

```python
pd.DataFrame(gym_info.agent.optim.loss).shape
```

```python
pd.DataFrame(gym_info.agent.optim.loss).mean(1).expanding().mean().plot(
    ylim=(0.09, 0.15)
)
```

```python
i_batch = 0
w = gym_info.agent.optim.updated_params["snarimax/~/linear"]["w"][:, i_batch, :, 0]
w = pd.DataFrame(w)

ax = w.plot(title=f"weights on batch {i_batch}")
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel("time")
ax.set_ylabel("weight value")


plt.figure()
w.iloc[-1][::-1].plot(kind="bar")
```

```python
w = gym_info.agent.optim.updated_params["snarimax/~/linear"]["w"].mean(axis=1)[:, :, 0]
w = pd.DataFrame(w)
ax = w.plot(title="averaged weights (over batches)")
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel("time")
ax.set_ylabel("weight")

plt.figure()
w.iloc[-1][::-1].plot(kind="bar")
```

## Taking mean inside simulation

```python
%%time


def add_batch(fun, take_mean=True):
    def fun_batch(*args, **kwargs):
        res = VMap(fun)(*args, **kwargs)
        if take_mean:
            res = jax.tree_map(lambda x: x.mean(axis=0), res)
        return res

    return fun_batch


gym_loop_batch = add_batch(gym_loop)
sim = unroll_transform_with_state(gym_loop_batch)

rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (10000, 20)) * 0.3

params, state = sim.init(rng, eps)
(gym_output, gym_info), final_state = sim.apply(params, state, rng, eps)
```

```python
pd.Series(-gym_output.reward).expanding().mean().plot(ylim=(0.09, 0.15))
```

```python
w = gym_info.agent.optim.updated_params["snarimax/~/linear"]["w"][:, :, 0]
w = pd.DataFrame(w)
ax = w.plot(title="averaged weights (over batches)")
ax.legend(bbox_to_anchor=(1.0, 1.0))
ax.set_xlabel("time")
ax.set_ylabel("weight")

plt.figure()
w.iloc[-1][::-1].plot(kind="bar")
```

# Hyper parameter tuning


## First order optimizers

We will consider different first order optimizers, namely:
- SGD
- ADAM
- ADAGRAD

For each of them, we will scan the "step_size" parameter $\eta$.

We will average results over batches of size 40.

We will consider trajectories of size 10.000.

Finally, we will pickup the best parameter based of the minimum averaged loss for the last 5000 time steps.

```python
%%time

STEP_SIZE_idx = pd.Index(onp.logspace(-4, 1, 30), name="step_size")
STEP_SIZE = jax.device_put(STEP_SIZE_idx.values)
OPTIMIZERS = [optax.sgd, optax.adam, optax.adagrad]

res = {}
for optimizer in tqdm(OPTIMIZERS):

    def gym_loop_scan_hparams(eps):
        def scan_params(step_size):
            return GymFeedback(build_agent(opt=optimizer(step_size)), env)(eps)

        res = VMap(scan_params)(STEP_SIZE)
        return res

    sim = unroll_transform_with_state(add_batch(gym_loop_scan_hparams))
    rng = jax.random.PRNGKey(42)
    eps = jax.random.normal(rng, (10000, 40)) * 0.3

    params, state = sim.init(rng, eps)
    _res, state = sim.apply(params, state, rng, eps)
    res[optimizer.__name__] = _res

```

```python
ax = None
BEST_STEP_SIZE = {}
BEST_GYM = {}
for name, (gym, info) in res.items():

    loss = pd.DataFrame(-gym.reward, columns=STEP_SIZE).iloc[-5000:].mean()

    BEST_STEP_SIZE[name] = loss.idxmin()
    best_idx = loss.reset_index(drop=True).idxmin()
    BEST_GYM[name] = jax.tree_map(lambda x: x[:, best_idx], gym)

    ax = loss[loss < 0.15].plot(logx=True, logy=False, ax=ax, label=name)
plt.legend()
```

```python
for name, gym in BEST_GYM.items():
    ax = (
        pd.Series(-gym.reward)
        .expanding()
        .mean()
        .plot(
            label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(0.09, 0.15)
        )
    )
ax.legend(bbox_to_anchor=(1.0, 1.0))
```

```python
pd.Series(BEST_STEP_SIZE).plot(kind="bar", logy=True)
```


## Newton algorithm


Now let's consider the newton algorithm.


First let's test it with one set of parameter with average over 20 batches.

```python
%%time


@add_batch
def gym_loop_newton(eps):
    return GymFeedback(build_agent(opt=newton(0.05, eps=20.0)), env)(eps)


sim = unroll_transform_with_state(gym_loop_newton)
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (10000, 20)) * 0.3

params, state = sim.init(rng, eps)
(gym, info), state = sim.apply(params, state, rng, eps)
```


```python
pd.Series(-gym.reward).expanding().mean().plot()
```

```python
%%time

STEP_SIZE = pd.Index(onp.logspace(-2, 3, 10), name="step_size")
EPS = pd.Index(onp.logspace(-4, 3, 5), name="eps")

HPARAMS_idx = pd.MultiIndex.from_product([STEP_SIZE, EPS])
HPARAMS = jnp.stack(list(map(onp.array, HPARAMS_idx)))


@add_batch
def gym_loop_scan_hparams(eps):
    def scan_params(hparams):
        step_size, newton_eps = hparams
        agent = build_agent(opt=newton(step_size, eps=newton_eps))
        return GymFeedback(agent, env)(eps)

    return VMap(scan_params)(HPARAMS)


sim = unroll_transform_with_state(gym_loop_scan_hparams)
rng = jax.random.PRNGKey(42)
eps = jax.random.normal(rng, (10000, 40)) * 0.3

params, state = sim.init(rng, eps)
res_newton, state = sim.apply(params, state, rng, eps)
```


```python
gym_newton, info_newton = res_newton
```

```python
loss_newton = pd.DataFrame(-gym_newton.reward, columns=HPARAMS_idx).mean().unstack()
```


```python
loss_newton = (
    pd.DataFrame(-gym_newton.reward, columns=HPARAMS_idx).iloc[-5000:].mean().unstack()
)
```


```python
sns.heatmap(loss_newton[loss_newton < 0.4], annot=True, cmap="YlGnBu")

STEP_SIZE, NEWTON_EPS = loss_newton.stack().idxmin()

x = -gym_newton.reward[-5000:].mean(axis=0)
x = jax.ops.index_update(x, jnp.isnan(x), jnp.inf)
I_BEST_PARAM = jnp.argmin(x)


BEST_NEWTON_GYM = jax.tree_map(lambda x: x[:, I_BEST_PARAM], gym_newton)
print("Best newton parameters: ", STEP_SIZE, NEWTON_EPS)
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).rolling(5000, min_periods=5000).mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(0.09, 0.1)
    )

gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .rolling(5000, min_periods=5000)
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Rolling mean of loss (5000) time-steps")
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).expanding().mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(0.09, 0.15)
    )
gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .expanding()
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
```


In agreement with Hazan's paper, we see that Newton's algorithm performs much better than sgd.

In addition, we note that
- adam does not perform well in this online setting.
- adagrad performormance is between newton and sgd.


# Non-stationnary environment


```python
def scan_hparams_first_order():

    STEP_SIZE_idx = pd.Index(onp.logspace(-4, 1, 30), name="step_size")
    STEP_SIZE = jax.device_put(STEP_SIZE_idx.values)
    OPTIMIZERS = [optax.sgd, optax.adam, optax.adagrad]

    res = {}
    for optimizer in tqdm(OPTIMIZERS):

        def gym_loop_scan_hparams(eps):
            def scan_params(step_size):
                return GymFeedback(build_agent(opt=optimizer(step_size)), env)(eps)

            res = VMap(scan_params)(STEP_SIZE)
            return res

        sim = unroll_transform_with_state(add_batch(gym_loop_scan_hparams))

        params, state = sim.init(rng, eps)
        _res, state = sim.apply(params, state, rng, eps)
        res[optimizer.__name__] = _res

    ax = None
    BEST_STEP_SIZE = {}
    BEST_GYM = {}
    for name, (gym, info) in res.items():

        loss = pd.DataFrame(-gym.reward, columns=STEP_SIZE).iloc[-5000:].mean()

        BEST_STEP_SIZE[name] = loss.idxmin()
        best_idx = loss.reset_index(drop=True).idxmin()
        BEST_GYM[name] = jax.tree_map(lambda x: x[:, best_idx], gym)

        ax = loss[loss < 0.15].plot(logx=True, logy=False, ax=ax, label=name)
    plt.legend()
    return BEST_STEP_SIZE, BEST_GYM
```

```python
def scan_hparams_newton():
    STEP_SIZE = pd.Index(onp.logspace(-2, 3, 10), name="step_size")
    EPS = pd.Index(onp.logspace(-4, 3, 5), name="eps")

    HPARAMS_idx = pd.MultiIndex.from_product([STEP_SIZE, EPS])
    HPARAMS = jnp.stack(list(map(onp.array, HPARAMS_idx)))

    @add_batch
    def gym_loop_scan_hparams(eps):
        def scan_params(hparams):
            step_size, newton_eps = hparams
            agent = build_agent(opt=newton(step_size, eps=newton_eps))
            return GymFeedback(agent, env)(eps)

        return VMap(scan_params)(HPARAMS)

    sim = unroll_transform_with_state(gym_loop_scan_hparams)
    params, state = sim.init(rng, eps)
    res_newton, state = sim.apply(params, state, rng, eps)

    gym_newton, info_newton = res_newton

    loss_newton = pd.DataFrame(-gym_newton.reward, columns=HPARAMS_idx).mean().unstack()

    loss_newton = (
        pd.DataFrame(-gym_newton.reward, columns=HPARAMS_idx)
        .iloc[-5000:]
        .mean()
        .unstack()
    )

    sns.heatmap(loss_newton[loss_newton < 0.4], annot=True, cmap="YlGnBu")

    STEP_SIZE, NEWTON_EPS = loss_newton.stack().idxmin()

    x = -gym_newton.reward[-5000:].mean(axis=0)
    x = jax.ops.index_update(x, jnp.isnan(x), jnp.inf)
    I_BEST_PARAM = jnp.argmin(x)

    BEST_NEWTON_GYM = jax.tree_map(lambda x: x[:, I_BEST_PARAM], gym_newton)
    print("Best newton parameters: ", STEP_SIZE, NEWTON_EPS)
    return (STEP_SIZE, NEWTON_EPS), BEST_NEWTON_GYM
```

# Setting 2


let's build an environment corresponding to "setting 2" in [1]

```python
from wax.modules import Counter

T = int(1.0e4)


def build_env_setting_2():
    def env(action, obs):
        y_pred, eps = action, obs
        t = Counter()()
        ar_coefs_1 = jnp.array([-0.4, -0.5, 0.4, 0.4, 0.1])
        ar_coefs_2 = jnp.array([0.6, -0.4, 0.4, -0.5, 0.5])
        alpha = t / T
        ar_coefs = ar_coefs_1 * alpha + ar_coefs_2 * (1 - alpha)
        ma_coefs = jnp.array([0.32, -0.2])

        y = ARMA(ar_coefs, ma_coefs)(eps)
        # prediction used on a fresh y observation.
        rw = -((y - y_pred) ** 2)

        env_info = {"y": y, "y_pred": y_pred}
        obs = y
        return rw, obs, env_info

    return env
```

## sample noise

```python
env = build_env_setting_2()
rng = jax.random.PRNGKey(42)

eps = jax.random.uniform(rng, (T, 20), minval=-0.5, maxval=0.5)
MIN_ERR = 0.0833
```


```python
BEST_STEP_SIZE, BEST_GYM = scan_hparams_first_order()
```

```python
(STEP_SIZE, NEWTON_EPS), BEST_NEWTON_GYM = scan_hparams_newton()
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).rolling(5000, min_periods=5000).mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(MIN_ERR, 0.1)
    )

gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .rolling(5000, min_periods=5000)
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Rolling mean of loss (5000) time-steps")
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).expanding().mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(MIN_ERR, 0.15)
    )
gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .expanding()
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
```


## Sensibility to hyper parameters

If we choose the wrong hyper-parameters, we see that the agent can failed to capture the changing dynamic.

```python
%%time


@add_batch
def gym_loop_newton(eps):
    return GymFeedback(build_agent(opt=newton(0.1, eps=0.3)), env)(eps)


sim = unroll_transform_with_state(gym_loop_newton)
params, state = sim.init(rng, eps)
(gym, info), state = sim.apply(params, state, rng, eps)

pd.Series(-gym.reward).expanding().mean().plot()  # ylim=(MIN_ERR, 0.12))
```


# Setting 3


let's build an environment corresponding to "setting 3" in [1]

```python
from wax.modules import Counter

T = int(1.0e4)


def build_env_setting_3():
    def env(action, obs):
        y_pred, eps = action, obs
        t = Counter()()
        ar_coefs_1 = jnp.array([0.6, -0.5, 0.4, -0.4, 0.3])
        ar_coefs_2 = jnp.array([-0.4, -0.5, 0.4, 0.4, 0.1])

        ar_coefs = jnp.where(t < T / 2, ar_coefs_1, ar_coefs_2)
        ma_coefs_1 = jnp.array([0.3, -0.2])
        ma_coefs_2 = jnp.array([-0.3, 0.2])
        ma_coefs = jnp.where(t < T / 2, ma_coefs_1, ma_coefs_2)

        y = ARMA(ar_coefs, ma_coefs)(eps)
        # prediction used on a fresh y observation.
        rw = -((y - y_pred) ** 2)

        env_info = {"y": y, "y_pred": y_pred}
        obs = y
        return rw, obs, env_info

    return env
```

## sample noise

```python
env = build_env_setting_3()
rng = jax.random.PRNGKey(42)

eps = jax.random.uniform(rng, (T, 20), minval=-0.5, maxval=0.5)
MIN_ERR = 0.0833
```


```python
BEST_STEP_SIZE, BEST_GYM = scan_hparams_first_order()
```

```python
(STEP_SIZE, NEWTON_EPS), BEST_NEWTON_GYM = scan_hparams_newton()
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).rolling(5000, min_periods=5000).mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(MIN_ERR, 0.1)
    )

gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .rolling(5000, min_periods=5000)
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Rolling mean of loss (5000) time-steps")
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).expanding().mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}", ylim=(MIN_ERR, 0.15)
    )
gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .expanding()
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
```


## Sensibility to hyper parameters

If we choose the wrong hyper-parameters, we see that the agent can failed to capture the changing dynamic.

```python
%%time


@add_batch
def gym_loop_newton(eps):
    return GymFeedback(build_agent(opt=newton(0.1, eps=0.3)), env)(eps)


sim = unroll_transform_with_state(gym_loop_newton)
params, state = sim.init(rng, eps)
(gym, info), state = sim.apply(params, state, rng, eps)

pd.Series(-gym.reward).expanding().mean().plot()  # ylim=(MIN_ERR, 0.12))
```


# Setting 4


let's build an environment corresponding to "setting 4" in [1]

```python
from wax.modules import Counter

T = int(1.0e4)


def build_env_setting_3():
    def env(action, obs):
        y_pred, eps = action, obs
        t = Counter()()
        ar_coefs = jnp.array([0.11, -0.5])

        ma_coefs = jnp.array([0.41, -0.39, -0.685, 0.1])

        rng = hk.next_rng_key()

        prev_eps = hk.get_state("prev_eps", (1,), init=lambda *_: jnp.zeros_like(eps))
        eps = prev_eps + eps  # jax.random.normal(rng, (1, 20))

        hk.set_state("prev_eps", eps)

        y = ARMA(ar_coefs, ma_coefs)(eps)
        # prediction used on a fresh y observation.
        rw = -((y - y_pred) ** 2)

        env_info = {"y": y, "y_pred": y_pred}
        obs = y
        return rw, obs, env_info

    return env
```


## sample noise

```python
env = build_env_setting_3()
rng = jax.random.PRNGKey(42)

eps = jax.random.normal(rng, (T, 20)) * 0.3
MIN_ERR = 0.3 ** 2
MAX_ERR = 0.22
```


```python
BEST_STEP_SIZE, BEST_GYM = scan_hparams_first_order()
```

```python
(STEP_SIZE, NEWTON_EPS), BEST_NEWTON_GYM = scan_hparams_newton()
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).rolling(5000, min_periods=5000).mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}",
        ylim=(MIN_ERR, MAX_ERR),
    )

gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .rolling(5000, min_periods=5000)
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
plt.title("Rolling mean of loss (5000) time-steps")
```

```python
for name, gym in BEST_GYM.items():
    pd.Series(-gym.reward).expanding().mean().plot(
        label=f"{name}    -    $\eta$={BEST_STEP_SIZE[name]:.2e}",
        ylim=(MIN_ERR, MAX_ERR),
    )
gym = BEST_NEWTON_GYM
ax = (
    pd.Series(-gym.reward)
    .expanding()
    .mean()
    .plot(
        label=f"Newton    -    $\eta$={STEP_SIZE:.2e},    $\epsilon$={NEWTON_EPS:.2e}"
    )
)
ax.legend(bbox_to_anchor=(1.0, 1.0))
```


As noted in [1], the newton algorithm seems to be the only one to achieve an average error rate that converges to the variance of the noise (0.09).



## Sensibility to hyper parameters



We observe that for too low value of the step size $\eta$,
the newton algorithm does not work anymore.

```python
%%time


@add_batch
def gym_loop_newton(eps):
    return GymFeedback(build_agent(opt=newton(0.18, eps=0.3)), env)(eps)


sim = unroll_transform_with_state(gym_loop_newton)
params, state = sim.init(rng, eps)
(gym, info), state = sim.apply(params, state, rng, eps)

pd.Series(-gym.reward).expanding().mean().plot()  # ylim=(MIN_ERR, 0.12))

```
