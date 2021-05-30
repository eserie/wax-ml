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
"""Format nested data structures to numpy/xarray/pandas containers."""
from typing import Any, TypeVar

import numpy as onp
import xarray as xr
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten

DatasetCoordinates = TypeVar(
    "DatasetCoordinates"
)  # xr.core.coordinates.DatasetCoordinates


def _to_dataarray(ref_coords, dataarray, dims):
    coords = {
        ref_dim: ref_coord
        for ref_dim, ref_coord in ref_coords.items()
        if ref_dim in dims
    }
    assert isinstance(
        dataarray, onp.ndarray
    ), "[format_to_dataarray] dataarray should have been converted to numpy array."

    dataarray = xr.DataArray(
        dataarray,
        dims=dims,
        coords=coords,
    )
    return dataarray


def _to_dataframe(ref_coords, dataarray, dims, index_nlevels=1):
    dataarray = _to_dataarray(ref_coords, dataarray, dims)
    if dataarray.ndim == 1:
        return dataarray.to_series()
    elif dataarray.ndim == 2:
        return dataarray.to_pandas()
    else:
        series = dataarray.to_series()
        idx_columns = (onp.arange(index_nlevels, series.index.nlevels)).tolist()
        return series.unstack(idx_columns)


def _to_series(ref_coords, dataarray, dims):
    dataarray = _to_dataarray(ref_coords, dataarray, dims)
    series = dataarray.to_series()
    return series


def _get_dims_from_data(data):
    shape = onp.shape(data)
    dims = tuple(f"dim_{i}" for i in range(len(shape)))
    return dims


def format_dataarray(
    ref_coords: DatasetCoordinates, data: Any, format_dims: Any
) -> Any:
    """Format data following a given data schema

    Args:
        ref_coords : coordinates referential to set data coordinates from format_dims.
        data : Nested data structure with numpy arrays as leaves.
        format_dims : Nested data structure with same treedef than data containing
            dims specification for convertion of leaves to DataArray.

    Returns:
        Nested data structure with same treedef as input data and leaves converted in DataArrays.
    """
    data_flat, treedef = tree_flatten(data)
    format_dims_flat, treedef_format = tree_flatten(format_dims)
    if treedef == treedef_format:
        format_dims_flat = tree_leaves(format_dims_flat)
        vals = tuple(
            _to_dataarray(ref_coords, dataarray, dims)
            for dataarray, dims in zip(data_flat, format_dims_flat)
        )
    else:
        vals = tuple(
            _to_dataarray(ref_coords, dataarray, format_dims) for dataarray in data_flat
        )

    return tree_unflatten(treedef, vals)


def format_dataset(ref_coords: DatasetCoordinates, data: Any, format_dims: Any) -> Any:
    """Format data following a given data schema

    Args:
        ref_coords : coordinates referential to set data coordinates from format_dims.
        data : Nested data structure with numpy arrays as leaves.
        format_dims : Nested data structure with same treedef than data containing
            dims specification for convertion of leaves to DataArray.

    Returns:
        Nested data structure with same treedef as input data and leaves converted in DataArrays.
    """
    output = format_dataarray(ref_coords, data, format_dims)
    try:
        return xr.Dataset(output)
    except TypeError:
        return output


def format_dataframe(
    ref_coords: DatasetCoordinates,
    data: Any,
    format_dims: Any = None,
    index_nlevels: int = 1,
) -> Any:
    """Format data following a given data schema

    Args:
        ref_coords : coordinates referential to set data coordinates from format_dims.
        data : Nested data structure with numpy arrays as leaves.
        format_dims : Nested data structure with same treedef than data containing
            dims specification for convertion of leaves to DataArray.
        index_nlevels : number of levels expected for output index.

    Returns:
        Nested data structure with same treedef as input data and leaves converted in DataArrays.
    """
    ref_coords = {} if ref_coords is None else ref_coords
    data_flat, treedef = tree_flatten(data)
    format_dims_flat, treedef_format = tree_flatten(format_dims)
    if treedef == treedef_format:
        vals = tuple(
            _to_dataframe(ref_coords, dataarray, dims, index_nlevels)
            for dataarray, dims in zip(data_flat, format_dims_flat)
        )
    else:
        if not format_dims:
            format_dims = _get_dims_from_data(data_flat[0])
        vals = tuple(
            _to_dataframe(ref_coords, dataarray, format_dims, index_nlevels)
            for dataarray in data_flat
        )

    return tree_unflatten(treedef, vals)


def format_series(ref_coords: DatasetCoordinates, data: Any, format_dims: Any) -> Any:
    """Format data following a given data schema

    Args:
        ref_coords : coordinates referential to set data coordinates from format_dims.
        data : Nested data structure with numpy arrays as leaves.
        format_dims : Nested data structure with same treedef than data containing
            dims specification for convertion of leaves to DataArray.

    Returns:
        Nested data structure with same treedef as input data and leaves converted in DataArrays.
    """
    data_flat, treedef = tree_flatten(data)
    format_dims_flat, treedef_format = tree_flatten(format_dims)
    if treedef == treedef_format:
        vals = tuple(
            _to_series(ref_coords, dataarray, dims)
            for dataarray, dims in zip(data_flat, format_dims_flat)
        )
    else:
        vals = tuple(
            _to_series(ref_coords, dataarray, format_dims) for dataarray in data_flat
        )
    return tree_unflatten(treedef, vals)
