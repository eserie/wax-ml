# Copyright 2014-2019, xarray Developers
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
"""Generate fake temperature data for tests purposes."""
import numpy as onp
import pandas as pd
import xarray as xr


def get_lats_lons_times(n_lat=25, n_lon=53, n_time=124, freq="H"):
    lats = onp.linspace(0, 20, n_lat)
    lons = onp.linspace(0, 11, n_lon)
    times = pd.date_range(start="2000/01/01", freq=freq, periods=n_time)
    return lats, lons, times


def generate_data(n_lat=25, n_lon=53, n_time=124, freq="H"):
    onp.random.seed(3)
    lats, lons, times = get_lats_lons_times(n_lat, n_lon, n_time, freq=freq)

    data = onp.random.rand(n_lon, n_lat, n_time)

    data = xr.DataArray(
        data,
        coords=[lons, lats, times],
        dims=["lon", "lat", "time"],
        name="data",
    ).transpose("time", "lon", "lat")
    return data


def generate_times_data(n_lat=25, n_lon=53, n_time=124, freq="H"):
    onp.random.seed(3)

    lats, lons, times = get_lats_lons_times(n_lat, n_lon, n_time, freq=freq)

    times_arr = onp.random.choice(times, size=(n_lon, n_lat, n_time))
    times_data = xr.DataArray(
        times_arr,
        coords=[lons, lats, times],
        dims=["lon", "lat", "time"],
        name="data",
    ).transpose("time", "lon", "lat")
    return times_data


def generate_temperature_data():
    da = xr.Dataset()
    da["air"] = generate_data()
    return da


def generate_temperature_data_multi_time_scale():
    da = xr.Dataset()
    da["air"] = generate_data(n_time=12, freq="d").rename({"time": "day"})
    da["ground"] = generate_data(freq="H")
    return da
