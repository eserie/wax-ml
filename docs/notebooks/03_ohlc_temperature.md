---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,py,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
# Uncomment to run the notebook in Colab
# ! pip install -q "wax-ml[complete]@git+https://github.com/eserie/wax-ml.git"
# ! pip install -q --upgrade jax jaxlib==0.1.70+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

```python
# check available devices
import jax
```

```python
print("jax backend {}".format(jax.lib.xla_bridge.get_backend().platform))
jax.devices()
```

# 🌡 Binning temperatures 🌡

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/03_ohlc_temperature.ipynb)


Let's again considering the air temperatures dataset.
It is sampled at an hourly resolution.
We will make "trailing" air temperature bins during each day and "reset" the bin
aggregation process at each day change.

```python
import numpy as onp
import xarray as xr
```

```python
from wax.accessors import register_wax_accessors
from wax.modules import OHLC, HasChanged

register_wax_accessors()
```

```python
dataset = xr.tutorial.open_dataset("air_temperature")
dataset["date"] = dataset.time.dt.date.astype(onp.datetime64)
```

```python
dataset
```

```python
def bin_temperature(da):
    day_change = HasChanged()(da["date"])
    return OHLC()(da["air"], reset_on=day_change)


output, state = dataset.wax.stream().apply(
    bin_temperature, format_dims=onp.array(dataset.air.dims)
)
output = xr.Dataset(output._asdict())
```

```python
df = output.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().loc["2013-01"]
_ = df.plot(figsize=(12, 8), title="Trailing Open-High-Low-Close temperatures")
```

## The `UpdateOnEvent` module

The `OHLC` module uses the primitive `wax.modules.UpdateOnEvent`.

Its implementation required to complete Haiku with a central function
`set_params_or_state_dict` which we have actually integrated in this WAX-ML module.

We have opened an [issue on the Haiku github](https://github.com/deepmind/dm-haiku/issues/126)
to integrate it in Haiku.
