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

# # 〰 Compute exponential moving averages with xarray and pandas accessors 〰
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/01_demo_EWMA.ipynb)

# WAX-ML implements pandas and xarray accessors to ease the usage of machine-learning algorithms with
# high-level data APIs :
# - pandas's `DataFrame` and `Series`
# - xarray's `Dataset` and `DataArray`.
#
# These accessors allow to easily execute any function using Haiku modules
# on these data containers.
#
# For instance, WAX-ML propose an implementation of the exponential moving average realized
# with this mechanism.
#
# Let's show how it works.

# ## Load accessors

# First you need to load accessors:

# + tags=[]
from wax.accessors import register_wax_accessors

register_wax_accessors()
# -

# ## EWMA on dataframes

# Let's look at a simple example: The exponential moving average (EWMA).
#
# Let's apply the EWMA algorithm to the [NCEP/NCAR 's Air temperature data](http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html).

# ### 🌡 Load temperature dataset 🌡

# + tags=[]
import xarray as xr

dataset = xr.tutorial.open_dataset("air_temperature")
# -

# Let's see what this dataset looks like:

# + tags=[]
dataset
# -

# To compute a EWMA on some variables of a dataset, we usually need to convert data
# in pandas
# [series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) or
# [dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).
#
# So, let's convert the dataset into a dataframe to illustrate `accessors` on a dataframe:

dataframe = dataset.air.to_series().unstack(["lon", "lat"])

# ### EWMA with pandas

air_temp_ewma = dataframe.ewm(alpha=1.0 / 10.0).mean()
_ = air_temp_ewma.iloc[:, 0].plot()

# ### EWMA with WAX-ML

air_temp_ewma = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
_ = air_temp_ewma.iloc[:, 0].plot()

# On small data, WAX-ML's EWMA is slower than Pandas' because of the expensive data conversion steps.
# WAX-ML's accessors are interesting to use on large data loads
# (See our [three-steps_workflow](https://wax-ml.readthedocs.io/en/latest/notebooks/04_The_three_steps_workflow.html))
#
# ## Apply a custom function to a Dataset

# Now let's illustrate how WAX-ML accessors work on [xarray datasets](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html).

# +
from wax.modules import EWMA


def my_custom_function(dataset):
    return {
        "air_10": EWMA(1.0 / 10.0)(dataset["air"]),
        "air_100": EWMA(1.0 / 100.0)(dataset["air"]),
    }


dataset = xr.tutorial.open_dataset("air_temperature")
output, state = dataset.wax.stream().apply(
    my_custom_function, format_dims=dataset.air.dims
)

_ = output.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
