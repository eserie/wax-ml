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

# # ⏱ Synchronize data streams ⏱
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/02_Synchronize_data_streams.ipynb)

# Physicists, and not the least 😅, have brought a solution to the synchronization
# problem.  See [Poincaré-Einstein synchronization Wikipedia
# page](https://en.wikipedia.org/wiki/Einstein_synchronisation) for more details.
#
# In WAX-ML we strive to follow their recommendations and implement a synchronization
# mechanism between different data streams. Using the terminology of Henri Poincaré (see
# link above), we introduce the notion of "local time" to unravel the stream in which
# the user wants to apply transformations. We call the other streams "secondary streams".
# They can work at different frequencies, lower or higher.  The data from these secondary
# streams will be represented in the "local time" either with the use of a
# forward filling mechanism for lower frequencies or a buffering mechanism
# for higher frequencies.
#
# We implement a "data tracing" mechanism to optimize access to out-of-sync streams.
# This mechanism works on in-memory data.  We perform the first pass on the data,
# without actually accessing it, and determine the indices necessary to
# later access the data. Doing so we are vigilant to not let any "future"
# information pass through and thus guaranty a data processing that respects causality.
#
# The buffering mechanism used in the case of higher frequencies works with a fixed
# buffer size (see the WAX-ML module
# [`wax.modules.Buffer`](https://wax-ml.readthedocs.io/en/latest/_autosummary/wax.modules.buffer.html#module-wax.modules.buffer))
# which allows us to use JAX / XLA optimizations and have efficient processing.
#
# Let's illustrate with a small example how `wax.stream.Stream` synchronizes data streams.
#
# Let's use the dataset "air temperature" with :
# - An air temperature is defined with hourly resolution.
# - A "fake" ground temperature is defined with a daily resolution as the air temperature minus 10 degrees.

# + tags=[]
import xarray as xr

dataset = xr.tutorial.open_dataset("air_temperature")
dataset["ground"] = dataset.air.resample(time="d").last().rename({"time": "day"}) - 10
# -

# Let's see what this dataset looks like:

dataset

# + tags=[]
from wax.accessors import register_wax_accessors

register_wax_accessors()

# + tags=[]
from wax.modules import EWMA


def my_custom_function(dataset):
    return {
        "air_10": EWMA(1.0 / 10.0)(dataset["air"]),
        "air_100": EWMA(1.0 / 100.0)(dataset["air"]),
        "ground_100": EWMA(1.0 / 100.0)(dataset["ground"]),
    }


# -

results, state = dataset.wax.stream(
    local_time="time", ffills={"day": 1}, pbar=True
).apply(my_custom_function, format_dims=dataset.air.dims)

_ = results.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
