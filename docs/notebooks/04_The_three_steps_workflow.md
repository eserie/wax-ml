---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: ipynb,py,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
# Uncomment to run the notebook in Colab
# ! pip install -q "wax-ml[complete]@git+https://github.com/eserie/wax-ml.git"
# ! pip install -q --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

```{code-cell} ipython3
# check available devices
import jax
```

```{code-cell} ipython3
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
jax.devices()
```

# 🎛 The 3-steps workflow 🎛

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)

+++

It is already very useful to be able to execute a JAX function on a dataframe in a single work step
and with a single command line thanks to WAX-ML accessors.

The 1-step WAX-ML's stream API works like that:
```python
<data-container>.stream(...).apply(...)
```

But this is not optimal because, under the hood, there are mainly three costly steps:
- (1) (synchronize | data tracing | encode): make the data "JAX ready"
- (2) (compile | code tracing | execution): compile and optimize a function for XLA, execute it.
- (3) (format): convert data back to pandas/xarray/numpy format.

With the `wax.stream` primitives, it is quite easy to explicitly split the 1-step workflow
into a 3-step workflow.

This will allow the user to have full control over each step and iterate on each one.

It is actually very useful to iterate on step (2), the "calculation step" when
you are doing research.
You can then take full advantage of the JAX primitives, especially the `jit` primitive.

Let's illustrate how to reimplement WAX-ML EWMA yourself with the WAX-ML 3-step workflow.

+++

## Imports

```{code-cell} ipython3
:tags: []

import haiku as hk
import numpy as onp
import pandas as pd
import xarray as xr

from wax.accessors import register_wax_accessors
from wax.compile import jit_init_apply
from wax.external.eagerpy import convert_to_tensors
from wax.format import format_dataframe
from wax.modules import EWMA
from wax.stream import tree_access_data
from wax.unroll import dynamic_unroll

register_wax_accessors()
```

## Performance on big dataframes

+++

### Generate data

```{code-cell} ipython3
:tags: [parameters]

T = 1.0e5
N = 1000
```

```{code-cell} ipython3
:tags: []

%%time
T, N = map(int, (T, N))
dataframe = pd.DataFrame(
    onp.random.normal(size=(T, N)), index=pd.date_range("1970", periods=T, freq="s")
)
```

### Pandas EWMA

```{code-cell} ipython3
:tags: []

%%time
df_ewma_pandas = dataframe.ewm(alpha=1.0 / 10.0).mean()
```

### WAX-ML EWMA

```{code-cell} ipython3
:tags: []

%%time
df_ewma_wax = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
```

It's a little faster, but not that much faster...

+++

### WAX-ML EWMA (without format step)

+++

Let's disable the final formatting step (the output is now in raw JAX format):

```{code-cell} ipython3
:tags: []

%%time
df_ewma_wax_no_format = dataframe.wax.ewm(alpha=1.0 / 10.0, format_outputs=False).mean()
```

```{code-cell} ipython3
type(df_ewma_wax_no_format)
```

Let's check the device on which the calculation was performed (if you have GPU available, this should be `GpuDevice` otherwise it will be `CpuDevice`):

```{code-cell} ipython3
df_ewma_wax_no_format.device()
```

That's better! In fact (see below)
there is a performance problem in the final formatting step.
See WEP3 for a proposal to improve the formatting step.

+++

### Generate data (in dataset format)

WAX-ML `Sream` object works on datasets.
So let's transform the `DataFrame` into a xarray `Dataset`:

```{code-cell} ipython3
dataset = xr.DataArray(dataframe).to_dataset(name="dataarray")
```

## Step (1) (synchronize | data tracing | encode)

In this step,  WAX-ML do:
- "data tracing" : prepare the indices for fast access tin the JAX function `access_data`
- synchronize streams if there is multiple ones.
  This functionality have options : `freq`, `ffills`
- encode and convert data from numpy to JAX: use encoders for `datetimes64` and `string_`
  dtypes. Be aware that by default JAX works in float32
  (see [JAX's Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision) to work in float64).

We have a function `Stream.prepare` that implement this Step (1).
It prepares a function that wraps the input function with the actual data and indices
in a pair of pure functions (`TransformedWithState` Haiku tuple).

```{code-cell} ipython3
%%time
stream = dataframe.wax.stream()
```

Define our custom function to be applied on a dict of arrays
having the same structure than the original dataset:

```{code-cell} ipython3
def my_ewma_on_dataset(dataset):
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])
```

```{code-cell} ipython3
transform_dataset, jxs = stream.prepare(dataset, my_ewma_on_dataset)
```

Let's definite the init parameters and state of the transformation we
will apply.

+++

### Init params and state

```{code-cell} ipython3
from wax.unroll import init_params_state
```

```{code-cell} ipython3
rng = jax.random.PRNGKey(42)
params, state = init_params_state(transform_dataset, rng, jxs)
```

```{code-cell} ipython3
params
```

```{code-cell} ipython3
assert state["ewma"]["count"].shape == (N,)
assert state["ewma"]["mean"].shape == (N,)
```

## Step (2) (compile | code tracing | execution)

+++

In this step we:
- prepare a pure function (with
  [Haiku's transform mechanism](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku-transforms))
  Define a "transformation" function which:
    - access to the data
    - apply another transformation, here: EWMA

- compile it with `jax.jit`
- perform code tracing and execution (the last line):
    - Unroll the transformation on "steps" `xs` (a `np.arange` vector).

```{code-cell} ipython3
rng = next(hk.PRNGSequence(42))
outputs, state = dynamic_unroll(transform_dataset, params, state, rng, False, jxs)
```

```{code-cell} ipython3
outputs.device()
```

Once it has been compiled and "traced" by JAX, the function is much faster to execute:

```{code-cell} ipython3
:tags: []

%%timeit
outputs, state = dynamic_unroll(transform_dataset, params, state, rng, False, jxs)
```

```{code-cell} ipython3
%%time
outputs, state = dynamic_unroll(transform_dataset, params, state, rng, False, jxs)
```

This is 3x faster than pandas implementation!

(The 3x factor is obtained by measuring the execution with %timeit.
We don't know why, but when executing a code cell once at a time, then the execution time can vary a lot and we can observe some executions with a speed-up of 100x).

+++

### Manually prepare the data and manage the device

+++

In order to manage the device on which the computations take place,
we need to have even more control over the execution flow.
Instead of calling `stream.prepare` to build the `transform_dataset` function,
we can do it ourselves by :
- using the `stream.trace_dataset` function
- converting the numpy data in jax ourself
- puting the data on the device we want.

```{code-cell} ipython3
np_data, np_index, xs = stream.trace_dataset(dataset)
jnp_data, jnp_index, jxs = convert_to_tensors((np_data, np_index, xs), "jax")
```

We explicitly set data on CPUs (the is not needed if you only have CPUs):

```{code-cell} ipython3
from jax.tree_util import tree_leaves, tree_map

cpus = jax.devices("cpu")
jnp_data, jnp_index, jxs = tree_map(
    lambda x: jax.device_put(x, cpus[0]), (jnp_data, jnp_index, jxs)
)
print("data copied to CPU device.")
```

We have now "JAX-ready" data for later fast access.

+++

Let's define the transformation that wrap the actual data and indices in a pair of
pure functions:

```{code-cell} ipython3
%%time
@jit_init_apply
@hk.transform_with_state
def transform_dataset(step):
    dataset = tree_access_data(jnp_data, jnp_index, step)
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])
```

And we can call it as before:

```{code-cell} ipython3
%%time
outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
```

```{code-cell} ipython3
outputs.device()
```

## Step(3) (format)
Let's come back to pandas/xarray:

```{code-cell} ipython3
%%time
y = format_dataframe(
    dataset.coords, onp.array(outputs), format_dims=dataset.dataarray.dims
)
```

It's quite slow (see WEP3 enhancement proposal).

+++

## GPU execution

+++

## GPU execution

+++

Let's look with execution on GPU

```{code-cell} ipython3
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
```

Let's check that our data is on the GPUs:

```{code-cell} ipython3
tree_leaves(jnp_data)[0].device()
```

```{code-cell} ipython3
tree_leaves(jnp_index)[0].device()
```

```{code-cell} ipython3
jxs.device()
```

```{code-cell} ipython3
%%time
if GPU_AVAILABLE:
    rng = next(hk.PRNGSequence(42))
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
```

Let's redefine our function `transform_dataset` by explicitly specify to `jax.jit` the `device` option.

```{code-cell} ipython3
%%time
if GPU_AVAILABLE:

    @hk.transform_with_state
    def transform_dataset(step):
        dataset = tree_access_data(jnp_data, jnp_index, step)
        return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])

    transform_dataset = type(transform_dataset)(
        transform_dataset.init, jax.jit(transform_dataset.apply, device=gpus[0])
    )

    rng = next(hk.PRNGSequence(42))
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
```

```{code-cell} ipython3
outputs.device()
```

```{code-cell} ipython3
%%timeit
if GPU_AVAILABLE:
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
```
