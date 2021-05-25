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

# 🎛 The 3-steps workflow 🎛

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)

+++

It is already very useful to be able to execute a JAX function on a dataframe in a single work step
and with a single command line thanks to WAX accessors.

The 1-step WAX's stream API works like that:
```python
.stream(...).apply(...)
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

Let's illustrate how to reimplement WAX EWMA yourself with the WAX 3-step workflow.

+++

## Imports

```{code-cell} ipython3
:tags: []

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
```

## Performance on big dataframes

+++

### Generate data

```{code-cell} ipython3
:tags: []

%%time
T = int(1.0e6)
N = 1000

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

### WAX EWMA

```{code-cell} ipython3
:tags: []

%%time
df_ewma_wax = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
```

It's a little faster, but not that much faster...

+++

### WAX EWMA (without format step)

+++

Let's disable the final formatting step (the output is now in raw JAX format):

```{code-cell} ipython3
:tags: []

%%time
df_ewma_wax_no_format = dataframe.wax.ewm(alpha=1.0 / 10.0, format_outputs=False).mean()
```

That's better! In fact (see below)
there is a performance problem in the final formatting step.
See WEP3 for a proposal to improve the formatting step.

+++

### Generate data (in dataset format)

WAX `Sream` object works on datasets to we'll move form dataframe to datasets.

```{code-cell} ipython3
dataset = xr.DataArray(dataframe).to_dataset(name="dataarray")
```

## Step (1) (synchronize | data tracing | encode)

In this step,  WAX do:
- "data tracing" : prepare the indices for fast access tin the JAX function `access_data`
- synchronize streams if there is multiple ones.
  This functionality have options : `freq`, `ffills`
- encode and convert data from numpy to JAX: use encoders for `datetimes64` and `string_`
  dtypes. Be aware that by default JAX works in float32
  (see [JAX's Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision) to work in float64).

```{code-cell} ipython3
%%time
stream = dataframe.wax.stream()
np_data, np_index, xs = stream.trace_dataset(dataset)
jnp_data, jnp_index, jxs = convert_to_tensors((np_data, np_index, xs), "jax")
```

We have now "JAX-ready" data for later fast access.

+++

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
%%time
@jit_init_apply
@hk.transform_with_state
def transform_dataset(step):
    dataset = partial(tree_access_data, jnp_data, jnp_index)(step)
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])


rng = next(hk.PRNGSequence(42))
outputs, state = dynamic_unroll(transform_dataset, xs, rng)
```

Once it has been compiled and "traced" by JAX, the function is much faster to execute:

```{code-cell} ipython3
:tags: []

%%time
outputs, state = dynamic_unroll(transform_dataset, xs, rng)
```

This is between 10x and 40x faster (time may vary) and 130 x faster than pandas implementation!

+++

## Step(3) (format)
Let's come back to pandas/xarray:

```{code-cell} ipython3
%%time
y = format_dataframe(
    dataset.coords, onp.array(outputs), format_dims=dataset.dataarray.dims
)
```

It's quite slow (see WEP3 enhancement proposal).
