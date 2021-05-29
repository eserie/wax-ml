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

# ⏱ Synchronize data streams ⏱

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/02_Synchronize_data_streams.ipynb)

In Colab install wax by executing this line in a cell:
```python
! pip install "wax-ml[dev,complete] @ git+https://github.com/eserie/wax-ml.git"
```

+++

Let's illustrate with a small example how `wax.stream.Stream` synchronizes data streams.

Let's use the dataset "air temperature" with :
- An air temperature defined with hourly resolution.
- A "fake" ground temperature defined with a daily resolution as the air temperature minus 10 degrees.

```{code-cell} ipython3
:tags: []

import xarray as xr

da = xr.tutorial.open_dataset("air_temperature")
da["ground"] = da.air.resample(time="d").last().rename({"time": "day"}) - 10
```

Let's see what this dataset looks like:

```{code-cell} ipython3
da
```

```{code-cell} ipython3
:tags: []

from wax.accessors import register_wax_accessors

register_wax_accessors()
```

```{code-cell} ipython3
:tags: []

from wax.modules import EWMA


def my_custom_function(da):
    return {
        "air_10": EWMA(1.0 / 10.0)(da["air"]),
        "air_100": EWMA(1.0 / 100.0)(da["air"]),
        "ground_100": EWMA(1.0 / 100.0)(da["ground"]),
    }
```

```{code-cell} ipython3
results, state = da.wax.stream(master_time="time", pbar=True).apply(
    my_custom_function, format_dims=da.air.dims
)
```

```{code-cell} ipython3
_ = results.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
```
