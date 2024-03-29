---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python colab={"base_uri": "https://localhost:8080/"} id="2b295407-92c4-4818-bfb9-f445f6967f10" outputId="dc6c8e1b-2875-4287-d83a-4bdc4c9db80a"
# Uncomment to run the notebook in Colab
# ! pip install -q "wax-ml[complete]@git+https://github.com/eserie/wax-ml.git"
# ! pip install -q --upgrade jax jaxlib==0.1.70+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

```python id="ff30291d"
# check available devices
import jax
```

```python colab={"base_uri": "https://localhost:8080/"} id="a3cdb104" outputId="a6f395c3-6ee3-4fe5-ce39-a02617a129ca"
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
jax.devices()
```

<!-- #region id="1fa1808c" -->
# 🎛 The 3-steps workflow 🎛

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)
<!-- #endregion -->

<!-- #region id="2e1eefec" -->
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
<!-- #endregion -->

<!-- #region id="bd2e906e" -->
## Imports
<!-- #endregion -->

```python id="2bdfbf9a"
import numpy as onp
import pandas as pd
import xarray as xr

from wax.accessors import register_wax_accessors
from wax.format import format_dataframe
from wax.modules import EWMA
from wax.stream import tree_access_data
from wax.unroll import unroll

register_wax_accessors()
```

<!-- #region id="a0999c69" -->
## Performance on big dataframes
<!-- #endregion -->

<!-- #region id="6f8447b4" -->
### Generate data
<!-- #endregion -->

```python id="768d7802-580d-4c31-9a0e-e6dc4f0589ca" tags=["parameters"]
T = 1.0e5
N = 1000
```

```python id="03af743d"
T, N = map(int, (T, N))
dataframe = pd.DataFrame(
    onp.random.normal(size=(T, N)), index=pd.date_range("1970", periods=T, freq="s")
)
```

<!-- #region id="d1fd46f7" -->
### pandas EWMA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="27092faf" outputId="e97a2738-5c2a-4bf9-c9e0-c44040185424"
%%time
df_ewma_pandas = dataframe.ewm(alpha=1.0 / 10.0).mean()
```

<!-- #region id="678be283" -->
### WAX-ML EWMA
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="11f3705d" outputId="744f0faa-7030-4155-80ae-4a5c745731f8"
%%time
df_ewma_wax = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
```

<!-- #region id="0d94d5cf" -->
It's a little faster, but not that much faster...
<!-- #endregion -->

<!-- #region id="e51ee290" -->
### WAX-ML EWMA (without format step)
<!-- #endregion -->

<!-- #region id="7361def9" -->
Let's disable the final formatting step (the output is now in raw JAX format):
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="f87f5668" outputId="3f37d97e-1875-4f66-9e6b-9a22b20642a6"
%%time
df_ewma_wax_no_format = dataframe.wax.ewm(alpha=1.0 / 10.0, format_outputs=False).mean()
df_ewma_wax_no_format.block_until_ready()
```

```python colab={"base_uri": "https://localhost:8080/"} id="88d0cab5-62ad-47f7-9d06-56fc45fa542e" outputId="f8b7c4fb-79a7-4652-82eb-2b649aeb074c"
type(df_ewma_wax_no_format)
```

<!-- #region id="9be62475-b2fa-4a95-b293-7e4410ca36ca" -->
Let's check the device on which the calculation was performed (if you have GPU available, this should be `GpuDevice` otherwise it will be `CpuDevice`):
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="38d3970c-04a5-4deb-93b3-a4f5f899e8f1" outputId="ba287d55-719a-47de-f959-9223effbe7a6"
df_ewma_wax_no_format.device()
```

<!-- #region id="784ee16e" -->
Now we will see how to break down WAX-ML one-liners `<dataset>.ewm(...).mean()` or `<dataset>.stream(...).apply(...)` into 3 steps:
- a preparation step where we prepare JAX-ready data and functions.
- a processing step where we execute the JAX program
- a post-processing step where we format the results in pandas or xarray format.
<!-- #endregion -->

<!-- #region id="c5e7b817" -->
### Generate data (in dataset format)

WAX-ML `Sream` object works on datasets.
So let's transform the `DataFrame` into a xarray `Dataset`:
<!-- #endregion -->

```python id="9965444b"
dataset = xr.DataArray(dataframe).to_dataset(name="dataarray")
del dataframe
```

<!-- #region id="123965eb" -->
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
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="5f6b72ca" outputId="e6852b1f-65bb-478a-ea6f-1cc6a3ce0898"
%%time
stream = dataset.wax.stream()
```

<!-- #region id="4cc34aad-3e15-4220-9dff-30dcad660307" -->
Define our custom function to be applied on a dict of arrays
having the same structure than the original dataset:
<!-- #endregion -->


```python id="2bfa2a6c"
def my_ewma_on_dataset(dataset):
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])
```

```python id="4735903f"
transform_dataset, jxs = stream.prepare(dataset, my_ewma_on_dataset)
```

<!-- #region id="18b31346-501a-407b-b586-780117d043f3" -->
Let's definite the init parameters and state of the transformation we
will apply.
<!-- #endregion -->


<!-- #region id="903a778c" -->
## Step (2) (compile | code tracing | execution)
<!-- #endregion -->

<!-- #region id="425830f4" -->
In this step we:
- prepare a pure function (with
  [Haiku's transform mechanism](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku-transforms))
  Define a "transformation" function which:
    - access to the data
    - apply another transformation, here: EWMA

- compile it with `jax.jit`
- perform code tracing and execution (the last line):
    - Unroll the transformation on "steps" `xs` (a `np.arange` vector).
<!-- #endregion -->

```python id="6825fdc8-1773-4e04-a41c-a126a1527891"
outputs = unroll(transform_dataset)(jxs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="3862a486-a5a2-4aa9-967a-59ebc32a18e1" outputId="b54b4e2c-a0ab-4a43-d331-d6b3d90707c6"
outputs.device()
```

<!-- #region id="b73f3252" -->
Once it has been compiled and "traced" by JAX, the function is much faster to execute:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a889d294" outputId="e69a75ff-4b00-4a1d-f101-49944e01d9b4"
%%timeit
outputs = unroll(transform_dataset)(jxs)
_ = outputs.block_until_ready()
```

<!-- #region id="987e8b63" -->
This is 3x faster than pandas implementation!
<!-- #endregion -->

<!-- #region id="4c8acec1-4414-4340-9cfc-199e90565d4d" -->
### Manually prepare the data and manage the device
<!-- #endregion -->

<!-- #region id="8b76775b-76f7-45ae-8bc7-66a6f945370f" -->
In order to manage the device on which the computations take place,
we need to have even more control over the execution flow.
Instead of calling `stream.prepare` to build the `transform_dataset` function,
we can do it ourselves by :
- using the `stream.trace_dataset` function
- converting the numpy data in jax ourself
- puting the data on the device we want.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="76479f76-7b6e-4321-97c3-942f2987bc01" outputId="7f22f580-be0b-4d23-9d4d-7d89c9dd117b"
np_data, np_index, xs = stream.trace_dataset(dataset)
jnp_data, jnp_index, jxs = jax.device_put((np_data, np_index, xs))
```

<!-- #region id="a4fa1934-3633-4661-b808-c60e4b8c4600" -->
We explicitly set data on CPUs (the is not needed if you only have CPUs):
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="bae591f7-e7ec-4b8b-9b80-6f129184e1ae" outputId="36f15d54-170f-4771-f571-ecacbc942026"
from jax.tree_util import tree_leaves, tree_map

cpus = jax.devices("cpu")
jnp_data, jnp_index, jxs = tree_map(
    lambda x: jax.device_put(x, cpus[0]), (jnp_data, jnp_index, jxs)
)
print("data copied to CPU device.")
```


<!-- #region id="f851fbe7-096e-4399-ae5c-08739837dfeb" -->
We have now "JAX-ready" data for later fast access.
<!-- #endregion -->

<!-- #region id="19f3b58e-61f3-481c-a512-cb87bde622a8" -->
Let's define the transformation that wrap the actual data and indices in a pair of
pure functions:
<!-- #endregion -->


```python id="e7ebbb08-d790-4977-b49d-c9224e299a42"
@jax.jit
@unroll
def transform_dataset(step):
    dataset = tree_access_data(jnp_data, jnp_index, step)
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])
```

<!-- #region id="31fa1d05-c3d7-4543-ac82-d398044cdc2e" -->
And we can call it as before:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="f851407a-597f-4711-8370-c3e83cb50da7" outputId="3892357f-2d91-40b6-85bc-542e471ed28f"
%%time
outputs = transform_dataset(jxs)
_ = outputs.block_until_ready()
```

```python colab={"base_uri": "https://localhost:8080/"} id="f851407a-597f-4711-8370-c3e83cb50da7" outputId="3892357f-2d91-40b6-85bc-542e471ed28f"
%%time
outputs = transform_dataset(jxs)
_ = outputs.block_until_ready()
```

```python colab={"base_uri": "https://localhost:8080/"} id="e5b786b7-ecaf-4280-894b-aa8f65a0b78f" outputId="2e3c126e-9c9d-49b4-caa5-67f215944188"
outputs.device()
```

<!-- #region id="6fa51498" -->
## Step(3) (format)
Let's come back to pandas/xarray:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d1b0732c" outputId="38e747c5-5b1c-40dc-c289-5aeebe503a8b"
%%time
y = format_dataframe(
    dataset.coords, onp.array(outputs), format_dims=dataset.dataarray.dims
)
```

<!-- #region id="e27bd3b8" -->
It's quite slow (see WEP3 enhancement proposal).
<!-- #endregion -->

<!-- #region id="3cdc9281-6092-4368-a0e8-ed26a5114106" -->
## GPU execution
<!-- #endregion -->

<!-- #region id="f0d4651d-9087-4f04-9f07-a4d92cd3ba1f" -->
Let's look with execution on GPU
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="a30f6024-5e9c-4174-92ea-7207860d829d" outputId="95058ee2-9461-4e7e-f9b1-b1e55087dc9e"
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

<!-- #region id="27d7cf26-f35b-489e-83b9-c120565d9b17" -->
Let's check that our data is on the GPUs:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="d6f00e5d-1b85-483d-9856-35de3a954b13" outputId="9d5d0ec1-4899-4d8e-d9f6-2232321ec42a"
tree_leaves(jnp_data)[0].device()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6f89fc7c-4824-4858-aee8-6fff7834c70c" outputId="5b5304de-0694-4944-92f1-63eee852ebf6"
tree_leaves(jnp_index)[0].device()
```

```python colab={"base_uri": "https://localhost:8080/"} id="ba2471ef-51e4-49f2-863c-aabafd401cbf" outputId="e6afa394-3e1b-4b9b-aea6-9dfe0d60ab86"
jxs.device()
```

```python colab={"base_uri": "https://localhost:8080/"} id="6c35bd5d-0113-455d-bd56-f7c31bf6c736" outputId="fda693e8-723e-436d-84f6-68a49a510be9"
%%time
if GPU_AVAILABLE:
    outputs = unroll(transform_dataset)(jxs)
```


<!-- #region id="274b5615-8785-43e7-a2db-5f867566c913" -->
Let's redefine our function `transform_dataset` by explicitly specify to `jax.jit` the `device` option.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="6bb7431e-90f2-4761-a906-69f25fea4a63" outputId="285660f2-8379-4b8f-f89f-9f5a7240c419"
%%time
from functools import partial

if GPU_AVAILABLE:

    @partial(jax.jit, device=gpus[0])
    @unroll
    def transform_dataset(step):
        dataset = tree_access_data(jnp_data, jnp_index, step)
        return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])

    outputs = transform_dataset(jxs)
```

```python colab={"base_uri": "https://localhost:8080/"} id="944dcb8e-211c-4b39-b854-12118fe775ed" outputId="fbeeab1e-e245-4d85-ad5c-09530ec7d331"
outputs.device()
```

```python colab={"base_uri": "https://localhost:8080/"} id="de99ee71-4e02-4843-9e6e-39d831f9697e" outputId="4262991e-c488-4296-849d-b111e288c203"
%%timeit
if GPU_AVAILABLE:
    outputs = unroll(transform_dataset)(jxs)
    _ = outputs.block_until_ready()
```
