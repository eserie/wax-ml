# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md:myst
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Uncomment to run the notebook in Colab
# # ! pip install -q "wax-ml[complete]@git+https://github.com/eserie/wax-ml.git"
# # ! pip install -q --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# -

# check available devices
import jax

print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
jax.devices()

# # 🎛 The 3-steps workflow 🎛
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)

# It is already very useful to be able to execute a JAX function on a dataframe in a single work step
# and with a single command line thanks to WAX-ML accessors.
#
# The 1-step WAX-ML's stream API works like that:
# ```python
# <data-container>.stream(...).apply(...)
# ```
#
# But this is not optimal because, under the hood, there are mainly three costly steps:
# - (1) (synchronize | data tracing | encode): make the data "JAX ready"
# - (2) (compile | code tracing | execution): compile and optimize a function for XLA, execute it.
# - (3) (format): convert data back to pandas/xarray/numpy format.
#
# With the `wax.stream` primitives, it is quite easy to explicitly split the 1-step workflow
# into a 3-step workflow.
#
# This will allow the user to have full control over each step and iterate on each one.
#
# It is actually very useful to iterate on step (2), the "calculation step" when
# you are doing research.
# You can then take full advantage of the JAX primitives, especially the `jit` primitive.
#
# Let's illustrate how to reimplement WAX-ML EWMA yourself with the WAX-ML 3-step workflow.

# ## Imports

# + tags=[]
from functools import partial

import haiku as hk
import numpy as onp
import pandas as pd
import xarray as xr
from eagerpy import convert_to_tensors

from wax.accessors import register_wax_accessors
from wax.compile import jit_init_apply
from wax.format import format_dataframe
from wax.modules import EWMA
from wax.stream import tree_access_data
from wax.unroll import dynamic_unroll

register_wax_accessors()
# -

# ## Performance on big dataframes

# ### Generate data

# + tags=["parameters"]
T = 1.0e5
N = 1000

# + tags=[]
# %%time
T, N = map(int, (T, N))
dataframe = pd.DataFrame(
    onp.random.normal(size=(T, N)), index=pd.date_range("1970", periods=T, freq="s")
)
# -

# ### Pandas EWMA

# + tags=[]
# %%time
df_ewma_pandas = dataframe.ewm(alpha=1.0 / 10.0).mean()
# -

# ### WAX-ML EWMA

# + tags=[]
# %%time
df_ewma_wax = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
# -

# It's a little faster, but not that much faster...

# ### WAX-ML EWMA (without format step)

# Let's disable the final formatting step (the output is now in raw JAX format):

# + tags=[]
# %%time
df_ewma_wax_no_format = dataframe.wax.ewm(alpha=1.0 / 10.0, format_outputs=False).mean()
# -

type(df_ewma_wax_no_format)

# Let's check the device on which the calculation was performed (if you have GPU available, this should be `GpuDevice` otherwise it will be `CpuDevice`):

df_ewma_wax_no_format.device()

# That's better! In fact (see below)
# there is a performance problem in the final formatting step.
# See WEP3 for a proposal to improve the formatting step.

# ### Generate data (in dataset format)
#
# WAX-ML `Sream` object works on datasets.
# So let's transform the `DataFrame` into a xarray `Dataset`:

dataset = xr.DataArray(dataframe).to_dataset(name="dataarray")

# ## Step (1) (synchronize | data tracing | encode)
#
# In this step,  WAX-ML do:
# - "data tracing" : prepare the indices for fast access tin the JAX function `access_data`
# - synchronize streams if there is multiple ones.
#   This functionality have options : `freq`, `ffills`
# - encode and convert data from numpy to JAX: use encoders for `datetimes64` and `string_`
#   dtypes. Be aware that by default JAX works in float32
#   (see [JAX's Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision) to work in float64).

# %%time
stream = dataframe.wax.stream()
np_data, np_index, xs = stream.trace_dataset(dataset)
jnp_data, jnp_index, jxs = convert_to_tensors((np_data, np_index, xs), "jax")

from jax.tree_util import tree_leaves, tree_map

# We explicitly set data on CPUs (the is not needed if you only have CPUs)
cpus = jax.devices("cpu")
jnp_data, jnp_index, jxs = tree_map(
    lambda x: jax.device_put(x, cpus[0]), (jnp_data, jnp_index, jxs)
)
print("data copied to CPU device.")


# We have now "JAX-ready" data for later fast access.

# ## Step (2) (compile | code tracing | execution)

# In this step we:
# - prepare a pure function (with
#   [Haiku's transform mechanism](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku-transforms))
#   Define a "transformation" function which:
#     - access to the data
#     - apply another transformation, here: EWMA
#
# - compile it with `jax.jit`
# - perform code tracing and execution (the last line):
#     - Unroll the transformation on "steps" `xs` (a `np.arange` vector).

# +
# %%time
@jit_init_apply
@hk.transform_with_state
def transform_dataset(step):
    dataset = partial(tree_access_data, jnp_data, jnp_index)(step)
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])


rng = next(hk.PRNGSequence(42))
outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
# -

outputs.device()

# Once it has been compiled and "traced" by JAX, the function is much faster to execute:

# + tags=[]
# %%timeit
outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
# -

# %%time
outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)

# This is 3x faster than pandas implementation!
#
# (The 3x factor is obtained by measuring the execution with %timeit.
# We don't know why, but when executing a code cell once at a time, then the execution time can vary a lot and we can observe some executions with a speed-up of 100x).

# ## Step(3) (format)
# Let's come back to pandas/xarray:

# %%time
y = format_dataframe(
    dataset.coords, onp.array(outputs), format_dims=dataset.dataarray.dims
)

# It's quite slow (see WEP3 enhancement proposal).

# ## GPU execution


# ## GPU execution

# Let's look with execution on GPU

try:
    gpus = jax.devices("gpu")
    jnp_data, jnp_index, jxs = tree_map(
        lambda x: jax.device_put(x, gpus[0]), (jnp_data, jnp_index, jxs)
    )
    print("data copied to GPU device.")
    GPU_AVAILABLE = True
except RuntimeError as err:
    print(err)
    GPU_AVAILABLE = False

# Let's check that our data is on the GPUs:

tree_leaves(jnp_data)[0].device()

tree_leaves(jnp_index)[0].device()

jxs.device()

# %%time
if GPU_AVAILABLE:
    rng = next(hk.PRNGSequence(42))
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)


# Let's redefine our function `transform_dataset` by explicitly specify to `jax.jit` the `device` option.

# %%time
if GPU_AVAILABLE:

    @hk.transform_with_state
    def transform_dataset(step):
        dataset = partial(tree_access_data, jnp_data, jnp_index)(step)
        return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])

    transform_dataset = type(transform_dataset)(
        transform_dataset.init, jax.jit(transform_dataset.apply, device=gpus[0])
    )

    rng = next(hk.PRNGSequence(42))
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)

outputs.device()

# %%timeit
if GPU_AVAILABLE:
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
