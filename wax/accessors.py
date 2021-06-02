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
"""Define accessors for xarray and pandas data containers."""
from dataclasses import dataclass, field
from typing import Any, Callable, Tuple, Union

import jax.numpy as jnp
import numpy as onp
import pandas as pd
import xarray as xr

from wax.format import format_dataarray, format_dataframe, format_dataset, format_series
from wax.stream import Stream

# DTypeLike = TypeVar("DTypeLike")
DTypeLike = str


@dataclass(frozen=True)
class StreamDataset(Stream):
    """
    Args:
        accessor : Dataset accessor
    """

    accessor: xr.Dataset = field(default_factory=xr.Dataset)

    def apply(
        self,
        module: Callable,
        params: Any = None,
        state: Any = None,
        rng: jnp.ndarray = None,
        skip_first: bool = False,
        format_dims: Union[Tuple, onp.ndarray] = (),
    ) -> Any:
        """Apply a module to a dataset.

        Args:
            module : callable being able to be transformed with Haiku transform_with_state.
            params: parameters for the module.
            state : state for the module.
            rng: random number generator key.
            skip_first : if true, first value of the sequence is not used in apply.
            format_dims : nested data structure with specification of dims for dataarray formatting.

        Return:
            Unroll results of the module formated as a nested data structure with dataarray leaves.
        """
        dataset = self.accessor._obj
        schema = self.get_dataset_schema(dataset)
        outputs = self.unroll_dataset(
            module, params, state, rng, skip_first, schema.encoders, dataset
        )
        if self.return_state:
            outputs, state = outputs
        if self.format_outputs:
            try:
                outputs = format_dataset(schema.coords, outputs, format_dims)
            except ValueError:
                # try with reduced dims
                format_dims = (format_dims[0],)
                outputs = format_dataset(schema.coords, outputs, format_dims)

        if self.return_state:
            return outputs, state
        else:
            return outputs


@dataclass(frozen=True)
class StreamDataArray(Stream):
    """
    Args:
        accessor : DataArray accessor
    """

    accessor: xr.Dataset = field(default_factory=xr.Dataset)

    def apply(
        self,
        module: Callable,
        params: Any = None,
        state: Any = None,
        rng: jnp.ndarray = None,
        skip_first: bool = False,
        format_dims: Any = None,
    ) -> Any:
        """Apply a module to a dataset.

        Args:
            module : callable being able to be transformed with Haiku transform_with_state.
            params: parameters for the module.
            state : state for the module.
            rng: random number generator key.
            skip_first : if true, first value of the sequence is not used in apply.
            format_dims : nested data structure with specification of dims for dataarray formatting.


        Return:
            Unroll results of the module formated as a nested data structure with dataarray leaves.
        """
        dataarray = self.accessor._obj
        dataset = xr.Dataset({"dataarray": dataarray})
        schema = self.get_dataset_schema(dataset)

        def module_dataset(dataset):
            array = dataset["dataarray"]
            return module(array)

        outputs = self.unroll_dataset(
            module_dataset, params, state, rng, skip_first, schema.encoders, dataset
        )
        if self.return_state:
            outputs, state = outputs

        if self.format_outputs:
            if format_dims is None:
                format_dims = dataarray.dims

            try:
                outputs = format_dataarray(schema.coords, outputs, format_dims)
            except ValueError:
                # try with reduced dims
                format_dims = (format_dims[0],)
                outputs = format_dataarray(schema.coords, outputs, format_dims)

        if self.return_state:
            return outputs, state
        else:
            return outputs


@dataclass(frozen=True)
class StreamDataFrame(Stream):
    """
    Args:
        accessor : DataArray accessor
    """

    accessor: xr.Dataset = field(default_factory=xr.Dataset)

    def apply(
        self,
        module: Callable,
        params: Any = None,
        state: Any = None,
        rng: jnp.ndarray = None,
        skip_first: bool = False,
        format_dims: Any = None,
    ) -> Any:
        """Apply a module to a dataset.

        Args:
            module : callable being able to be transformed with Haiku transform_with_state.
            params: parameters for the module.
            state : state for the module.
            rng: random number generator key.
            skip_first : if true, first value of the sequence is not used in apply.
            format_dims : nested data structure with specification of dims for dataarray formatting.

        Return:
            Unroll results of the module formated as a nested data structure with dataarray leaves.
        """
        dataframe = self.accessor._obj

        def _has_multi_index(dataframe):
            if dataframe.index.nlevels > 1:
                return True
            if dataframe.columns.nlevels > 1:
                return True
            return False

        if not _has_multi_index(dataframe):
            # avoid to stack the dataframe : its faster!
            dataarray = xr.DataArray(dataframe)
        else:
            dataarray = dataframe.stack(
                list(range(dataframe.columns.nlevels))
            ).to_xarray()
        dataset = dataarray.to_dataset(name="dataarray")
        schema = self.get_dataset_schema(dataset)

        def module_dataset(dataset):
            array = dataset["dataarray"]
            return module(array)

        outputs = self.unroll_dataset(
            module_dataset, params, state, rng, skip_first, schema.encoders, dataset
        )
        if self.return_state:
            outputs, state = outputs

        if self.format_outputs:
            if format_dims is None:
                format_dims = dataarray.dims

            try:
                outputs = format_dataframe(
                    schema.coords,
                    outputs,
                    format_dims,
                    index_nlevels=dataframe.index.nlevels,
                )
            except ValueError:
                # try with reduced dims
                format_dims = (format_dims[0],)
                outputs = format_dataframe(
                    schema.coords,
                    outputs,
                    format_dims,
                    index_nlevels=dataframe.index.nlevels,
                )

        if self.return_state:
            return outputs, state
        else:
            return outputs


@dataclass(frozen=True)
class StreamSeries(Stream):
    """
    Args:
        accessor : DataArray accessor
    """

    accessor: xr.Dataset = field(default_factory=xr.Dataset)

    def apply(
        self,
        module: Callable,
        params: Any = None,
        state: Any = None,
        rng: jnp.ndarray = None,
        skip_first: bool = False,
        format_dims: Any = None,
    ) -> Any:
        """Apply a module to a dataset.

        Args:
            module : callable being able to be transformed with Haiku transform_with_state.
            params: parameters for the module.
            state : state for the module.
            rng: random number generator key.
            skip_first : if true, first value of the sequence is not used in apply.
            format_dims : nested data structure with specification of dims for dataarray formatting.

        Return:
            Unroll results of the module formated as a nested data structure with dataarray leaves.
        """
        series = self.accessor._obj
        dataarray = series.to_xarray()
        dataset = xr.Dataset({"dataarray": dataarray})
        schema = self.get_dataset_schema(dataset)

        def module_dataset(dataset):
            array = dataset["dataarray"]
            return module(array)

        outputs = self.unroll_dataset(
            module_dataset, params, state, rng, skip_first, schema.encoders, dataset
        )
        if self.return_state:
            outputs, state = outputs

        if self.format_outputs:
            if format_dims is None:
                format_dims = dataarray.dims

            try:
                outputs = format_series(schema.coords, outputs, format_dims)
            except ValueError:
                # try with reduced dims
                format_dims = (format_dims[0],)
                outputs = format_series(schema.coords, outputs, format_dims)
        if self.return_state:
            return outputs, state
        else:
            return outputs


class WaxAccessor:
    def ewm(self, *args, **kwargs):
        return ExponentialMovingWindow(accessor=self, *args, **kwargs)


@dataclass(frozen=True)
class ExponentialMovingWindow:
    accessor: WaxAccessor
    alpha: float
    adjust: bool = True
    return_state: bool = False
    format_outputs: bool = True

    def mean(self):
        from wax.modules import EWMA

        def _apply_ema(
            accessor, alpha, adjust, params=None, state=None, *args, **kwargs
        ):
            return accessor.stream(*args, **kwargs).apply(
                lambda x: EWMA(alpha, adjust)(x),
                params=params,
                state=state,
                rng=None,
            )

        return _apply_ema(**self.__dict__)


class WaxDatasetAccessor(WaxAccessor):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def stream(self, *args, **kwargs):
        return StreamDataset(accessor=self, *args, **kwargs)


class WaxDataArrayAccessor(WaxAccessor):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def stream(self, *args, **kwargs):
        return StreamDataArray(accessor=self, *args, **kwargs)


class WaxDataFrameAccessor(WaxAccessor):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def stream(self, *args, **kwargs):
        return StreamDataFrame(accessor=self, *args, **kwargs)


class WaxSeriesAccessor(WaxAccessor):
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def stream(self, *args, **kwargs):
        return StreamSeries(accessor=self, *args, **kwargs)


def register_wax_accessors():
    if not hasattr(xr.Dataset, "wax"):
        xr.register_dataset_accessor("wax")(WaxDatasetAccessor)
    if not hasattr(xr.DataArray, "wax"):
        xr.register_dataarray_accessor("wax")(WaxDataArrayAccessor)
    if not hasattr(pd.DataFrame, "wax"):
        pd.api.extensions.register_dataframe_accessor("wax")(WaxDataFrameAccessor)
    if not hasattr(pd.Series, "wax"):
        pd.api.extensions.register_series_accessor("wax")(WaxSeriesAccessor)
