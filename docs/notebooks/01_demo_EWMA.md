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

# xarray and pandas accessors

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/01_demo_EWMA.ipynb)

In Colab install wax by executing this line in a cell:
```python
! pip install "wax-ml[dev,complete] @ git+https://github.com/eserie/wax-ml.git"
```

+++

WAX implements pandas and xarray "accessors" to ease the usage of machine-learning algorithms with
high-level data APIs :
- pandas's `DataFrame` and `Series` and
- xarray's `Dataset` and `DataArray`.

These accessors allow to easily execute any function using Haiku modules
on these data containers.

For instance, WAX propose an implementation of the exponential moving average realized
with this mechanism.

Let's show how it works.

+++

## Load accessors

+++

First you need to load accessors:

```{code-cell} ipython3
:tags: []

from wax.accessors import register_wax_accessors

register_wax_accessors()
```

## EWMA on dataframes

+++

Let's look at a simple example: The exponential moving average (EWMA).

Let's apply the EWMA algorithm to the [NCEP/NCAR 's Air temperature data](http://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis.html).

+++

### 🌡 Load temperature dataset 🌡

```{code-cell} ipython3
:tags: []

import xarray as xr

da = xr.tutorial.open_dataset("air_temperature")
```

Let's see what this dataset looks like:

```{code-cell} ipython3
:tags: []

da
```

To compute a EWMA on some variables of a dataset, we usually need to convert data
in pandas
[series](https://pandas.pydata.org/docs/reference/api/pandas.Series.html) or
[dataframe](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html).

So, let's convert the dataset into a dataframe to illustrate `accessors` on a dataframe:

```{code-cell} ipython3
dataframe = da.air.to_series().unstack(["lon", "lat"])
```

### EWMA with pandas

```{code-cell} ipython3
%%time
air_temp_ewma = dataframe.ewm(alpha=1.0 / 10.0).mean()
_ = air_temp_ewma.iloc[:, 0].plot()
```

### EWMA with WAX

```{code-cell} ipython3
%%time
air_temp_ewma = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
_ = air_temp_ewma.iloc[:, 0].plot()
```

On small data, WAX's EWMA is slower than Pandas' because of the expensive data conversion steps.
WAX's accessors are interesting to use on large data loads
(See our [three-steps_workflow](https://wax-ml.readthedocs.io/en/latest/notebooks/04_The_three_steps_workflow.html))

## Apply a custom function to a Dataset

+++

Now let's illustrate how WAX accessors work on [xarray datasets](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html).

```{code-cell} ipython3
from wax.modules import EWMA


def my_custom_function(da):
    return {
        "air_10": EWMA(1.0 / 10.0)(da["air"]),
        "air_100": EWMA(1.0 / 100.0)(da["air"]),
    }


da = xr.tutorial.open_dataset("air_temperature")
output, state = da.wax.stream().apply(my_custom_function, format_dims=da.air.dims)

_ = output.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
```
