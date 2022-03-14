# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Uncomment to run the notebook in Colab
# # ! pip install -q "wax-ml[complete]@git+https://github.com/eserie/wax-ml.git"
# # ! pip install -q --upgrade jax jaxlib==0.1.70+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# -

# %pylab inline
# %load_ext autoreload
# %autoreload 2

# +
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import pandas as pd
from jax.config import config

from wax.modules.ewma import EWMA
from wax.unroll import unroll_transform_with_state
# -

# check available devices
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
jax.devices()

# +
adjust = True
ignore_na = False
config.update("jax_enable_x64", True)

T = 20

x = jnp.full((T,), jnp.nan).at[0].set(1).at[10].set(-1)

rng = jax.random.PRNGKey(38)
x = jax.random.normal(rng, (T,))

x = jnp.full((T,), jnp.nan).at[2].set(1).at[10].set(-1)


@partial(unroll_transform_with_state, dynamic=True)
def fun(x):
    return EWMA(1 / 10, adjust=adjust, ignore_na=ignore_na, return_info=True)(x)


rng = jax.random.PRNGKey(42)
params, state = fun.init(rng, x)
(res, info), final_state = fun.apply(params, state, rng, x)


res = pd.DataFrame(onp.array(res))

ref_res = (
    pd.DataFrame(onp.array(x))
    .ewm(alpha=1 / 10, adjust=adjust, ignore_na=ignore_na)
    .mean()
)
# -
res

ref_res

info

pd.testing.assert_frame_equal(res, ref_res, atol=1.0e-6)


# ##  check gradient

# +
@jax.value_and_grad
def batch(params):
    (res, info), final_state = fun.apply(params, state, rng, x)
    return jnp.nanmean(res)


score, grad = batch(params)
assert not jnp.isnan(grad["ewma"]["logcom"])
score, grad
# -

# # Linear adjustement

# +
adjust = True
ignore_na = False
config.update("jax_enable_x64", True)

T = 20

x = jnp.full((T,), jnp.nan).at[0].set(1).at[10].set(-1)

rng = jax.random.PRNGKey(38)
x = jax.random.normal(rng, (T,))


# -


from wax.unroll import unroll

x = jnp.full((20,), jnp.nan).at[2].set(1).at[10].set(-1)
(res, info) = unroll(
    lambda x: EWMA(com=10, adjust="linear", ignore_na=True, return_info=True)(x)
)(x)
res = pd.DataFrame(onp.array(res))
pd.Series(info["com_eff"]).plot()


# +
# rng = jax.random.PRNGKey(42)
# x = jax.random.normal(rng, (100,)).at[30:50].set(jnp.nan)
# x = jnp.full((100,), jnp.nan).at[2].set(1).at[10].set(-1)


# +
x = (
    jnp.ones((100,))
    .at[0]
    .set(-1)
    .at[30:50]
    .set(-1)
    .at[40:50]
    .set(jnp.nan)
    .at[3:20]
    .set(jnp.nan)
)

(res, info) = unroll(
    lambda x: EWMA(com=10, adjust="linear", ignore_na=True, return_info=True)(x)
)(x)
res = pd.Series(onp.array(res))
pd.Series(info["com_eff"]).plot()
res.plot()


# +
x = (
    jnp.ones((100,))
    .at[0]
    .set(-1)
    .at[30:50]
    .set(-1)
    .at[40:45]
    .set(jnp.nan)
    .at[3:20]
    .set(jnp.nan)
)

(res, info) = unroll(
    lambda x: EWMA(com=10, adjust="linear", ignore_na=False, return_info=True)(x)
)(x)
res = pd.Series(onp.array(res))
pd.Series(info["com_eff"]).plot()
res.plot()


# +
x = (
    jnp.ones((100,))
    .at[0]
    .set(-1)
    .at[30:50]
    .set(-1)
    .at[40:50]
    .set(jnp.nan)
    .at[3:20]
    .set(jnp.nan)
)

(res, info) = unroll(
    lambda x: EWMA(com=10, adjust="linear", ignore_na=False, return_info=True)(x)
)(x)
res = pd.Series(onp.array(res))
pd.Series(info["com_eff"]).plot()
res.plot()


# +
x = (
    jnp.ones((100,))
    .at[0]
    .set(-1)
    .at[30:50]
    .set(-1)
    .at[40:50]
    .set(jnp.nan)
    .at[3:20]
    .set(jnp.nan)
)

(res, info) = unroll(
    lambda x: EWMA(
        com=10, adjust=False, ignore_na=False, return_info=True, initial_value=jnp.nan
    )(x)
)(x)
res = pd.Series(onp.array(res))
# pd.Series(info["com_eff"]).plot()
res.plot()
(res, info) = unroll(
    lambda x: EWMA(
        com=10, adjust=False, ignore_na=False, return_info=True, initial_value=0.0
    )(x)
)(x)
res = pd.Series(onp.array(res))
# pd.Series(info["com_eff"]).plot()
res.plot()
plt.legend(("init nan", "init 0"))


# +
x = (
    jnp.ones((100,))
    .at[0]
    .set(-1)
    .at[30:50]
    .set(-1)
    .at[40:50]
    .set(jnp.nan)
    .at[3:20]
    .set(jnp.nan)
)

(res, info) = unroll(
    lambda x: EWMA(com=10, adjust=True, ignore_na=False, return_info=True)(x)
)(x)
res = pd.Series(onp.array(res))
pd.Series(info["com_eff"]).plot()
res.plot()


# +
x = (
    jnp.ones((100,))
    .at[0]
    .set(-1)
    .at[30:50]
    .set(-1)
    .at[40:50]
    .set(jnp.nan)
    .at[3:20]
    .set(jnp.nan)
)

alpha = 1 / (1 + 10)
(res, info) = unroll(
    lambda x: EWMA(com=10, adjust=True, ignore_na=True, return_info=True)(x),
    dynamic=False,
)(x)
res = pd.Series(onp.array(res))
pd.Series(info["com_eff"]).plot()
# pd.Series(info["old_wt"]/alpha).plot()

res.plot()


# -


# # Exponential adjustement 

# +
@partial(unroll_transform_with_state, dynamic=True)
def fun(x):
    return EWMA(1 / 10, adjust=True, ignore_na=False, return_info=True)(x)


rng = jax.random.PRNGKey(42)
params, state = fun.init(rng, x)
(res, info), final_state = fun.apply(params, state, rng, x)


res = pd.DataFrame(onp.array(res))


c1 = pd.DataFrame(info["com_eff"])
c1.plot()
# -

# # More checks 

# +
adjust = False
ignore_na = False

adjust = True
ignore_na = False


def run():
    x = jnp.ones((30,), "float64").at[0].set(-1).at[5:20].set(jnp.nan)

    @partial(unroll_transform_with_state)
    def fun(x):
        return EWMA(1 / 10, adjust=adjust, ignore_na=ignore_na, return_info=True)(x)

    rng = jax.random.PRNGKey(42)
    params, state = fun.init(rng, x)
    (res, info), final_state = fun.apply(params, state, rng, x)
    res = pd.Series(onp.array(res))

    ref_res = (
        pd.Series(onp.array(x))
        .ewm(alpha=1 / 10, adjust=adjust, ignore_na=ignore_na)
        .mean()
        .values
    )

    df = pd.concat(
        [
            pd.Series(x),
            pd.Series(onp.array(ref_res)),
            pd.Series(onp.array(res)),
        ],
        axis=1,
        keys=["x", "pandas", "wax"],
    )

    return df

    df = pd.concat(
        [
            pd.Series(x),
            ref_res,
            res,
            pd.Series(onp.array(info["mean"])),
            pd.Series(info["norm"]),
            pd.Series(onp.array(info["mean"])) / ref_res,
        ],
        axis=1,
        keys=["x", "pandas", "wax", "wax-mean", "wax-norm", "pandas-norm"],
    )
    df.plot()


# -

adjust = False
ignore_na = False
df = run()
df.plot()

adjust = True
ignore_na = False
df = run()
df.plot()

adjust = False
ignore_na = True
df = run()
df.plot()

adjust = True
ignore_na = True
df = run()
df.plot()

adjust = "linear"
ignore_na = True
df = run()
df.plot()

adjust = "linear"
ignore_na = False
df = run()
df.plot()

# # Numba implementation 

from wax.modules.ewma_numba import ewma, init

# +
x = onp.ones((30,), "float64")
x[0] = onp.nan

x[1] = -1

x[5:20] = onp.nan
x = x.reshape(-1, 1)
state = init(x)


res, state = ewma(com=10, adjust="linear")(x, state)
pd.DataFrame(res).plot()
# -

# # Online pandas ewm 

# +
online_ewm = pd.DataFrame(x).ewm(10).online()
res_tot = online_ewm.mean()


data = pd.DataFrame(x)
online_ewm = data.iloc[:10].ewm(10).online()
res1 = online_ewm.mean()

res2 = online_ewm.mean(update=data.iloc[10:])

df = pd.concat([res_tot, res1, res2], keys=["tot", "res1", "res2"], axis=1)
df.plot()
df
# -


