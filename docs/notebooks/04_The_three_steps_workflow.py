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

# + id="2b295407-92c4-4818-bfb9-f445f6967f10"
# Uncomment to run the notebook in Colab
# # ! pip install -q "wax-ml[complete]@git+https://github.com/eserie/wax-ml.git"
# # ! pip install -q --upgrade jax jaxlib==0.1.67+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html

# + id="ff30291d"
# check available devices
import jax

# + colab={"base_uri": "https://localhost:8080/"} id="a3cdb104" outputId="2a8b84e5-1d90-4c06-bed5-1518bdfdc767"
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
jax.devices()

# + [markdown] id="1fa1808c"
# # 🎛 The 3-steps workflow 🎛
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)

# + [markdown] id="2e1eefec"
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

# + [markdown] id="bd2e906e"
# ## Imports

# + id="2bdfbf9a" tags=[]
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

# + [markdown] id="a0999c69"
# ## Performance on big dataframes

# + [markdown] id="6f8447b4"
# ### Generate data

# + id="768d7802-580d-4c31-9a0e-e6dc4f0589ca" tags=["parameters"]
T = 1.0e5
N = 1000

# + colab={"base_uri": "https://localhost:8080/"} id="03af743d" outputId="ae3f7e8f-7e2c-4ef1-a456-bd4d2f72893d" tags=[]
T, N = map(int, (T, N))
dataframe = pd.DataFrame(
    onp.random.normal(size=(T, N)), index=pd.date_range("1970", periods=T, freq="s")
)

# + [markdown] id="d1fd46f7"
# ### Pandas EWMA

# + colab={"base_uri": "https://localhost:8080/"} id="27092faf" outputId="760f16e5-bc4d-4bf1-fa1b-7749e1d2366f" tags=[]
# %%time
df_ewma_pandas = dataframe.ewm(alpha=1.0 / 10.0).mean()

# + [markdown] id="678be283"
# ### WAX-ML EWMA

# + colab={"base_uri": "https://localhost:8080/"} id="11f3705d" outputId="75921f06-a92c-4e1a-a7f9-49866659c232" tags=[]
# %%time
df_ewma_wax = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()

# + [markdown] id="0d94d5cf"
# It's a little faster, but not that much faster...

# + [markdown] id="e51ee290"
# ### WAX-ML EWMA (without format step)

# + [markdown] id="7361def9"
# Let's disable the final formatting step (the output is now in raw JAX format):

# + colab={"base_uri": "https://localhost:8080/"} id="f87f5668" outputId="7e3e0052-803a-48ac-ae9a-c2b68af8c575" tags=[]
# %%time
df_ewma_wax_no_format = dataframe.wax.ewm(alpha=1.0 / 10.0, format_outputs=False).mean()

# + colab={"base_uri": "https://localhost:8080/"} id="88d0cab5-62ad-47f7-9d06-56fc45fa542e" outputId="41fe33ca-4713-4a9d-a595-5c85696ff76f"
type(df_ewma_wax_no_format)

# + [markdown] id="9be62475-b2fa-4a95-b293-7e4410ca36ca"
# Let's check the device on which the calculation was performed (if you have GPU available, this should be `GpuDevice` otherwise it will be `CpuDevice`):

# + colab={"base_uri": "https://localhost:8080/"} id="38d3970c-04a5-4deb-93b3-a4f5f899e8f1" outputId="ae302feb-5e16-4a53-8aad-6dd12de7d4e8"
df_ewma_wax_no_format.device()

# + [markdown] id="784ee16e"
# That's better! In fact (see below)
# there is a performance problem in the final formatting step.
# See WEP3 for a proposal to improve the formatting step.

# + [markdown] id="c5e7b817"
# ### Generate data (in dataset format)
#
# WAX-ML `Sream` object works on datasets.
# So let's transform the `DataFrame` into a xarray `Dataset`:

# + id="9965444b"
dataset = xr.DataArray(dataframe).to_dataset(name="dataarray")

# + [markdown] id="123965eb"
# ## Step (1) (synchronize | data tracing | encode)
#
# In this step,  WAX-ML do:
# - "data tracing" : prepare the indices for fast access tin the JAX function `access_data`
# - synchronize streams if there is multiple ones.
#   This functionality have options : `freq`, `ffills`
# - encode and convert data from numpy to JAX: use encoders for `datetimes64` and `string_`
#   dtypes. Be aware that by default JAX works in float32
#   (see [JAX's Common Gotchas](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision) to work in float64).
#
# We have a function `Stream.prepare` that implement this Step (1).
# It prepares a function that wraps the input function with the actual data and indices
# in a pair of pure functions (`TransformedWithState` Haiku tuple).

# + colab={"base_uri": "https://localhost:8080/"} id="5f6b72ca" outputId="18f8761e-f198-476c-a3e0-06b6773d874a"
# %%time
stream = dataframe.wax.stream()

# + [markdown] id="4cc34aad-3e15-4220-9dff-30dcad660307"
# Define our custom function to be applied on a dict of arrays
# having the same structure than the original dataset:


# + id="2bfa2a6c"
def my_ewma_on_dataset(dataset):
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])


# + id="4735903f"
transform_dataset, jxs = stream.prepare(dataset, my_ewma_on_dataset)

# + [markdown] id="18b31346-501a-407b-b586-780117d043f3"
# Let's definite the init parameters and state of the transformation we
# will apply.

# + [markdown] id="78ea9eda-d0cc-4dba-8b2a-60fbf1dd41bd"
# ### Init params and state

# + id="47b1ebdf-ff32-4c2a-ae0b-51b88882328b"
from wax.unroll import init_params_state

# + id="ea6651f6-0db8-48f0-b578-013b2cb74272"
rng = jax.random.PRNGKey(42)
params, state = init_params_state(transform_dataset, rng, jxs)

# + colab={"base_uri": "https://localhost:8080/"} id="dc9a4bce-83d1-494b-b053-c6f9ebfb7d0c" outputId="4c4e08a8-5c03-468f-96bb-c0b10fdb1d79"
params

# + id="4dd83bef-ef68-4576-973f-594f18123944"
assert state["ewma"]["count"].shape == (N,)
assert state["ewma"]["mean"].shape == (N,)


# + [markdown] id="903a778c"
# ## Step (2) (compile | code tracing | execution)

# + [markdown] id="425830f4"
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

# + id="6825fdc8-1773-4e04-a41c-a126a1527891"
rng = next(hk.PRNGSequence(42))
outputs, state = dynamic_unroll(transform_dataset, params, state, rng, False, jxs)

# + colab={"base_uri": "https://localhost:8080/"} id="3862a486-a5a2-4aa9-967a-59ebc32a18e1" outputId="bb926b58-cb8e-44df-a38d-56e6fa758466"
outputs.device()

# + [markdown] id="b73f3252"
# Once it has been compiled and "traced" by JAX, the function is much faster to execute:

# + colab={"base_uri": "https://localhost:8080/"} id="a889d294" outputId="1bbcad84-5bc0-49d3-865f-685e75892793" tags=[]
# %%timeit
outputs, _ = dynamic_unroll(transform_dataset, params, state, rng, False, jxs)
_ = outputs.block_until_ready()

# + [markdown] id="987e8b63"
# This is 3x faster than pandas implementation!

# + [markdown] id="4c8acec1-4414-4340-9cfc-199e90565d4d"
# ### Manually prepare the data and manage the device

# + [markdown] id="8b76775b-76f7-45ae-8bc7-66a6f945370f"
# In order to manage the device on which the computations take place,
# we need to have even more control over the execution flow.
# Instead of calling `stream.prepare` to build the `transform_dataset` function,
# we can do it ourselves by :
# - using the `stream.trace_dataset` function
# - converting the numpy data in jax ourself
# - puting the data on the device we want.

# + colab={"base_uri": "https://localhost:8080/"} id="76479f76-7b6e-4321-97c3-942f2987bc01" outputId="c4c46c35-4e93-4f7b-8197-0520f8f0e1e0"
np_data, np_index, xs = stream.trace_dataset(dataset)
jnp_data, jnp_index, jxs = convert_to_tensors((np_data, np_index, xs), "jax")

# + [markdown] id="a4fa1934-3633-4661-b808-c60e4b8c4600"
# We explicitly set data on CPUs (the is not needed if you only have CPUs):

# + colab={"base_uri": "https://localhost:8080/"} id="bae591f7-e7ec-4b8b-9b80-6f129184e1ae" outputId="f073bab3-7f7b-4ed6-a5df-5266418d4072"
from jax.tree_util import tree_leaves, tree_map

cpus = jax.devices("cpu")
jnp_data, jnp_index, jxs = tree_map(
    lambda x: jax.device_put(x, cpus[0]), (jnp_data, jnp_index, jxs)
)
print("data copied to CPU device.")


# + [markdown] id="f851fbe7-096e-4399-ae5c-08739837dfeb"
# We have now "JAX-ready" data for later fast access.

# + [markdown] id="19f3b58e-61f3-481c-a512-cb87bde622a8"
# Let's define the transformation that wrap the actual data and indices in a pair of
# pure functions:

# + colab={"base_uri": "https://localhost:8080/"} id="e7ebbb08-d790-4977-b49d-c9224e299a42" outputId="c91ac27c-b671-4bad-8e0d-c9eb0464ae3c"
@jit_init_apply
@hk.transform_with_state
def transform_dataset(step):
    dataset = tree_access_data(jnp_data, jnp_index, step)
    return EWMA(alpha=1.0 / 10.0, adjust=True)(dataset["dataarray"])


# + [markdown] id="31fa1d05-c3d7-4543-ac82-d398044cdc2e"
# And we can call it as before:

# + colab={"base_uri": "https://localhost:8080/"} id="f851407a-597f-4711-8370-c3e83cb50da7" outputId="585097e9-6025-4f45-886e-261cbdba4014"
# %%time
outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
_ = outputs.block_until_ready()

# + colab={"base_uri": "https://localhost:8080/"} id="e5b786b7-ecaf-4280-894b-aa8f65a0b78f" outputId="c8bc9f41-7587-4482-c496-f40523f6840f"
outputs.device()

# + [markdown] id="6fa51498"
# ## Step(3) (format)
# Let's come back to pandas/xarray:

# + colab={"base_uri": "https://localhost:8080/"} id="d1b0732c" outputId="7427109b-7cfa-455c-af20-080b73b86182"
# %%time
y = format_dataframe(
    dataset.coords, onp.array(outputs), format_dims=dataset.dataarray.dims
)

# + [markdown] id="e27bd3b8"
# It's quite slow (see WEP3 enhancement proposal).

# + [markdown] id="3cdc9281-6092-4368-a0e8-ed26a5114106"
# ## GPU execution

# + [markdown] id="f0d4651d-9087-4f04-9f07-a4d92cd3ba1f"
# Let's look with execution on GPU

# + colab={"base_uri": "https://localhost:8080/"} id="a30f6024-5e9c-4174-92ea-7207860d829d" outputId="ee0cc84f-e332-485b-fddd-ee5ce42b6ce8"
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

# + [markdown] id="27d7cf26-f35b-489e-83b9-c120565d9b17"
# Let's check that our data is on the GPUs:

# + colab={"base_uri": "https://localhost:8080/"} id="d6f00e5d-1b85-483d-9856-35de3a954b13" outputId="db9396c3-b35d-4cdc-aaa3-eadddc9da31c"
tree_leaves(jnp_data)[0].device()

# + colab={"base_uri": "https://localhost:8080/"} id="6f89fc7c-4824-4858-aee8-6fff7834c70c" outputId="40c8a645-5771-40f0-ffab-0e5322d59c77"
tree_leaves(jnp_index)[0].device()

# + colab={"base_uri": "https://localhost:8080/"} id="ba2471ef-51e4-49f2-863c-aabafd401cbf" outputId="25800a2f-4152-4714-9ceb-ba396fcf1790"
jxs.device()

# + colab={"base_uri": "https://localhost:8080/"} id="6c35bd5d-0113-455d-bd56-f7c31bf6c736" outputId="86e2ff5d-3b7b-4b65-8202-0feebc27063d"
# %%time
if GPU_AVAILABLE:
    rng = next(hk.PRNGSequence(42))
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)


# + [markdown] id="274b5615-8785-43e7-a2db-5f867566c913"
# Let's redefine our function `transform_dataset` by explicitly specify to `jax.jit` the `device` option.

# + colab={"base_uri": "https://localhost:8080/"} id="6bb7431e-90f2-4761-a906-69f25fea4a63" outputId="405cc724-22e8-48d6-e722-d6cea8a90e49"
# %%time
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

# + colab={"base_uri": "https://localhost:8080/"} id="944dcb8e-211c-4b39-b854-12118fe775ed" outputId="d1f353b2-a7ed-4457-a26c-38ac4125aa65"
outputs.device()

# + colab={"base_uri": "https://localhost:8080/"} id="de99ee71-4e02-4843-9e6e-39d831f9697e" outputId="88a5e604-a722-43bf-d90c-49b11fde96b2"
# %%timeit
if GPU_AVAILABLE:
    outputs, state = dynamic_unroll(transform_dataset, None, None, rng, False, jxs)
    _ = outputs.block_until_ready()
