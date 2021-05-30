# Copyright 2021 The WAX-ML Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as onp
import xarray as xr
from jax.config import config

from wax.accessors import register_wax_accessors
from wax.datasets.generate_temperature_data import generate_temperature_data
from wax.modules.has_changed import HasChanged
from wax.modules.ohlc import OHLC


def test_ohlc_temperature():
    register_wax_accessors()
    config.update("jax_enable_x64", True)

    # da = xr.tutorial.open_dataset("air_temperature").sel(time="2013-01")
    da = generate_temperature_data()
    da["date"] = da.time.dt.floor("d").astype(onp.datetime64)
    assert len(set(da.date.values.tolist())) == 6
    assert len(set(da.time.values.tolist())) == 124

    def bin_temperature(da):
        date_has_changed = HasChanged()(da["date"])
        return OHLC()(da["air"], reset_on=date_has_changed)

    output, state = da.wax.stream().apply(bin_temperature, format_dims=da.air.dims)
    output = xr.Dataset(output._asdict())
    df = output.isel(lat=0, lon=0).drop_vars(["lat", "lon"]).to_array().to_pandas().T
    assert (df["HIGH"] >= df["LOW"]).all()
    assert (df["HIGH"] >= df["CLOSE"]).all()
    assert (df["HIGH"] >= df["OPEN"]).all()
    assert (df["CLOSE"] >= df["LOW"]).all()
    assert (df["OPEN"] >= df["LOW"]).all()
