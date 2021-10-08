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
import pandas as pd
import xarray as xr

from wax.format import tree_auto_format_with_shape


def test_tree_auto_format_with_shape():
    ds = xr.Dataset()
    ds["x"] = pd.DataFrame(
        onp.random.normal(size=(3, 4)),
        pd.Index(onp.arange(3), name="a"),
        pd.Index(onp.arange(4), name="b"),
    )
    ds["y"] = pd.DataFrame(
        onp.random.normal(size=(4, 5)),
        pd.Index(onp.arange(4), name="c"),
        pd.Index(onp.arange(5), name="d"),
    )

    z = tree_auto_format_with_shape(ds, onp.random.normal(size=(3, 4)))
    assert isinstance(z, xr.DataArray)
    assert z.dims == ("a", "b")

    z = tree_auto_format_with_shape(
        ds, onp.random.normal(size=(3, 4)), output_format="dataframe"
    )
    assert isinstance(z, pd.DataFrame)
    assert z.index.name, z.columns.name == ("a", "b")

    z = tree_auto_format_with_shape(ds, onp.random.normal(size=(3, 5)))
    assert isinstance(z, onp.ndarray)

    z = tree_auto_format_with_shape(ds, onp.random.normal(size=(4, 5)))
    assert isinstance(z, xr.DataArray)
    assert z.dims == ("c", "d")
