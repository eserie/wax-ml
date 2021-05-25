# Copyright 2021 The Wax Authors
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
"""Define `Stream` object."""
from collections import defaultdict, deque
from dataclasses import dataclass, field
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    cast,
)

import eagerpy as ep
import haiku as hk
import jax
import numpy as onp
import pandas as pd
import xarray as xr
from haiku import transform_with_state
from jax import numpy as jnp
from jax import tree_flatten, tree_leaves, tree_map, tree_unflatten
from jax.tree_util import tree_multimap
from tqdm.auto import tqdm

from wax.encode import (
    Encoder,
    datetime64_encoder,
    encode_dataset,
    floor_datetime,
    string_encoder,
)
from wax.unroll import dynamic_unroll
from wax.utils import get_unique_dtype, time_less

# DTypeLike = TypeVar("DTypeLike")
DTypeLike = str

EncoderMapping = Union[Dict[str, Callable[[Any], Encoder]], Dict[str, Encoder]]


DEFAULT_TYPES_ENCODERS = cast(
    EncoderMapping,
    {
        onp.dtype(str).type: string_encoder,
        onp.datetime64: datetime64_encoder,
    },
)
DTYPE_INIT_VALUES = {
    jnp.float32: jnp.nan,
    jnp.float64: jnp.nan,
    jnp.int64: -99999,
    jnp.int32: -99999,
    jnp.uint64: 0,
    jnp.uint32: 0,
    # numpy
    onp.float32: onp.nan,
    onp.float64: onp.nan,
    onp.int64: -99999,
    onp.int32: -99999,
    onp.uint64: 0,
    onp.uint32: 0,
    bool: False,
    onp.bool_: False,
}


class StreamObservation(NamedTuple):
    time: Any
    value: Any


class DatasetSchema(NamedTuple):
    coords: xr.core.coordinates.DatasetCoordinates
    encoders: Dict[str, Encoder]


def get_dataset_schema(
    xdata: xr.Dataset,
    types_encoders: EncoderMapping = None,
    check: bool = False,
) -> DatasetSchema:
    """Determine data schema of a Dataset.

    Args:
        xdata : xarray Dataset
        types_encoders : Mapping of used for different types.
        check : if true, check encoders by testing a round-trip
            conversion on detected types.

    Returns:
        Dataset schema
    """
    if types_encoders is None:
        types_encoders = DEFAULT_TYPES_ENCODERS
    encoders = {}
    np_data = {}
    np_data.update({dim: var.values for dim, var in xdata.items()})
    np_data.update({dim: var.values for dim, var in xdata.coords.items()})
    for dim, np_var in np_data.items():
        if np_var.dtype.type in types_encoders.keys():
            encoder = types_encoders[np_var.dtype.type]
            if callable(encoder):
                encoder = encoder(np_var)
            encoders[dim] = encoder
            if check:
                codes = encoder.encode(np_var)
                assert (encoder.decode(codes) == np_var).ravel().all()
    schema = DatasetSchema(coords=xdata.coords, encoders=encoders)
    return schema


def _is_verbose(verbose, time_dim):
    if isinstance(verbose, bool):
        return verbose

    if isinstance(verbose, (list, tuple)):
        if time_dim in verbose:
            return True
    return False


class GeneratorState(NamedTuple):
    output_values: Any
    stream_obs: Any


def get_time_dataset(dataset):
    """ """
    time_coords = get_dataset_time_coords(dataset)
    time_dataset = xr.Dataset()
    for time_dim in time_coords:
        # time_dataset[time_dim + "_index"] = dataset[time_dim]
        time_dataset[time_dim + "_index"] = xr.DataArray(
            onp.arange(len(dataset[time_dim])),
            dims=(time_dim,),
            coords={time_dim: dataset[time_dim].values},
        )

    return time_dataset


def get_dataset_index_from_stream_index(
    stream_index: Mapping[str, "StreamObservation"]
) -> xr.Dataset:
    """Convert stream_data index to dataset index.

     Args:
         stream_index : mapping of StreamObservation items with numpy index in "value" attribute.
    Returns:
         dataset with "step" coordinate and numpy index as variables.
            Buffered streams have an additional coordinate 'time_dim'+ '_buffer_index'
            construct buffers of data.
    """
    dataset = xr.Dataset()
    for time_dim, idx in stream_index.items():
        assert isinstance(idx, StreamObservation)
        idx_value = idx.value[time_dim + "_index"]
        if idx_value.ndim == 2:
            dataset[time_dim] = xr.DataArray(
                idx_value, dims=("step", time_dim + "_buffer_index")
            )
        else:
            assert idx_value.ndim == 1
            dataset[time_dim] = xr.DataArray(idx_value, dims="step")

    return dataset


def _get_unique_time_dim(dataset):
    time_coords = {
        dim for dim, vals in dataset.coords.items() if vals.dtype.type is onp.datetime64
    }
    assert len(time_coords) == 1
    return time_coords.pop()


def get_dataset_time_coords(dataset):
    time_coords = {
        dim for dim, vals in dataset.coords.items() if vals.dtype.type is onp.datetime64
    }
    assert len(time_coords)
    return time_coords


def split_dataset_from_time_dims(dataset):
    time_coords = get_dataset_time_coords(dataset)

    datasets = defaultdict(xr.Dataset)

    # create empty dataset with time coords
    for time_dim in time_coords:
        datasets[time_dim][time_dim] = dataset[time_dim]

    # fill datasets with the input dataset variables
    for var_dim, var in dataset.items():
        var_time_dims = time_coords.intersection(set(var.dims))
        if var_time_dims:
            assert len(var_time_dims) == 1, (
                f"variable {var_dim} has multiple time dimensions {var_time_dims}."
                "It should contain only one."
            )
            var_time_dim = var_time_dims.pop()
            datasets[var_time_dim][var_dim] = var

    return datasets


def get_dataset_index(
    dataset: xr.Dataset, time_dataset_index: xr.Dataset
) -> xr.Dataset:
    """Construct dataset of index from a dataset and a time_dataset_index.

    Args:
        dataset : original dataset
        time_datset_index : dataset of index from different timelines
            constructed with the original dataset.

    Returns:
        Dataset of index.
    """
    dataset_index = xr.Dataset()
    n_steps = len(next(iter(time_dataset_index.values())))
    for dim, var in dataset.items():
        if set(var.dims).intersection(time_dataset_index.keys()):
            time_dim = var.dims[0]
            assert time_dim in time_dataset_index, (
                f"'{time_dim}' does not seems to be a time "
                f"dimensions in {time_dataset_index.keys()}. "
                "For the moment, only time dimension as first dim is supported."
            )
            dataset_index[dim] = time_dataset_index[time_dim]
        else:
            if not onp.shape(var):
                dataset_index[dim] = xr.DataArray(onp.arange(n_steps), dims=("step",))
            else:
                values_atleast_1d = onp.atleast_1d(var.values)
                # grid = onp.indices(values_atleast_1d.shape)
                flat_idx = onp.arange(len(values_atleast_1d.ravel()))
                dataset_index[dim] = xr.DataArray(
                    onp.outer(onp.arange(n_steps), flat_idx),
                    dims=("step", dim + "_flat_idx"),
                )
    return dataset_index


def tree_buffer_transform(maxlen, fill_value, init_value):
    init_value, treedef = tree_flatten(init_value)
    fill_value = tree_leaves(fill_value)
    tree_len = len(init_value)
    assert len(fill_value) == tree_len
    buffer = [deque(maxlen=maxlen) for x in init_value]

    def transform(data):
        data = tree_leaves(data)
        assert len(data) == tree_len
        output = []
        for b, x, fv in zip(buffer, data, fill_value):
            b.append(x)
            output.append(deque_to_list(b, fv))
        return tree_unflatten(treedef, output)

    return transform


def deque_to_list(buffer: deque, fill_value: Any):
    """Convert buffer to list of length maxlen and replace missing
    observations with fill_value.

    Args:
        buffer_list : deque
        fill_value : fill_value to use
    """
    maxlen = cast(int, buffer.maxlen)
    buffer_list = list(buffer)

    x = buffer_list[0]
    if x is pd.NaT or x.dtype in [
        onp.dtype("datetime64[ns]"),
        onp.dtype("<M8[ns]"),
        onp.dtype("<M8[us]"),
    ]:
        if len(buffer_list) < maxlen:
            buffer_list = [pd.NaT] * (maxlen - len(buffer_list)) + buffer_list
        # return a deque to avoid flattening if later processing (like in stream_unroll)
        return deque(buffer_list)
    else:
        if len(buffer_list) < maxlen:
            buffer_list = [onp.full_like(buffer_list[0], fill_value)] * (
                maxlen - len(buffer_list)
            ) + buffer_list
        return onp.stack(buffer_list)


def unroll_stream(
    stream: Generator, skip_first: bool = False, pbar: Union[bool, Sequence[Any]] = True
) -> Any:
    """Unroll a generator by collecting its outputs in a list.

    Args:
        stream : generator
        skip_first : if true, skip first observation
        pbar : if true, activate progressbar.

    Yield:
        Nested data structure with list of collected outputs as leaves.
    """
    # init
    obs = next(stream)

    obs_flat, treedef = tree_flatten(obs)
    num_leaves = len(obs_flat)

    # stream_scan
    def _init_outputs():
        if skip_first:
            return [[]] * num_leaves
        else:
            return list(map(lambda x: [x], obs_flat))

    outputs = _init_outputs()

    if pbar:
        stream = tqdm(stream, desc="stream_unroll")

    for obs in stream:
        obs_flat = tree_leaves(obs)
        assert len(obs_flat) == num_leaves
        for y, x in zip(outputs, obs_flat):
            y.append(x)

    # stack outputs
    for i in range(num_leaves):
        outputs[i] = onp.stack(outputs[i])

    # transpose outputs
    return tree_unflatten(treedef, outputs)


def dataset_to_numpy(dataset):
    """Convert a xarray Dataset into a dict of numpy arrays."""
    dataset_numpy = {dim: var.values for dim, var in dataset.items()}
    return dataset_numpy


def np_select_item(np_dataset: Dict[str, Any], i: int, trace: bool = False):
    """Select item of a dataset.

    Args:
        np_dataset : dict of numpy arrays
        i : element position to access.
        trace : if true, replace the actual value in the output by its position.
    """
    if trace:
        # return only indices
        return {dim: onp.array(i) for dim, var in np_dataset.items()}
    else:
        # return real data
        return {dim: var[i] for dim, var in np_dataset.items()}


def access_data(step: int, x: Any, idx: Any):
    """Access to an array at a given step using a predifined access index.

    Args:
        step : step of the observation.
        x : data to access
        idx : predifined access index.
    """
    step_idx = idx[step]
    # output = x[step_idx]
    output = jnp.where(
        step_idx < 0,
        jnp.full_like(jnp.atleast_1d(x)[step_idx], DTYPE_INIT_VALUES[x.dtype.type]),
        jnp.atleast_1d(x)[step_idx],
    )
    return output


def tree_access_data(data, index, step):
    """Access to data from given indices.

    Args:
        data : nested data structure with arrays leaves.
        index : nested data structure with index leaves permitting
            to access the data at a given step.
        step : step on which to access.
    """
    return tree_multimap(partial(access_data, step), data, index)


@dataclass(frozen=True)
class Stream:
    """
    Args:
        master_time : dimension along which we want to iterate
        freqs : mapping of frequencies used to embed master_time in lower frequency streams.
        ffills : mapping of bool to ffill secondary streams.
        buffer_maxlen : mapping of int describing buffer size for secondary streams
            during one iteration of the main stream_data.
            This is needed to guaranty a known in advance data indexing scheme
            compatible with XLA compilation.
            If not specified, only the last observation of secondary streams is conserved.
        buffer_dtype_init_values : mapping of initial values to consider per dtype for
            Buffers initialization.
        verbose:  if true activate prints in the data tracing process.
        tensor_type : tensor type used for convertion of the data.
        trace : if true, perform data tracing returning only indexes instead of actual data.
        format_outputs : if true, format outputs in numpy array, otherwise return outputs
            in raw format (Jax tensors).
        return_state: if true, return state, otherwise only return unrolled outputs.
    """

    master_time: str = ""
    freqs: Dict[str, str] = field(default_factory=dict)
    ffills: Dict[str, bool] = field(default_factory=dict)
    buffer_maxlen: Dict[str, int] = field(default_factory=dict)
    buffer_dtype_init_values: Dict[DTypeLike, Any] = field(
        default_factory=lambda: DTYPE_INIT_VALUES
    )
    verbose: bool = False
    pbar: Union[bool, Sequence] = False
    tensor_type: str = "jax"
    trace: bool = True
    format_outputs: bool = True
    return_state: bool = True

    def unroll_dataset(
        self,
        dataset: xr.Dataset,
        module: Callable,
        encoders: EncoderMapping,
    ) -> Any:
        """Unroll a module onto a dataset.

        Args:
            dataset : dataset on which the transformation is applied
            module : callable being able to be transformed with Haiku transform_with_state.
            encoders : encoders used to encode numpy dtypes which are not supported by Jax.

        Return:
            Unroll results of the module formated as a nested data structure with dataarray leaves.
        """

        np_data, np_index, xs = self.trace_dataset(dataset)
        # encode values
        np_data = encode_dataset(encoders, np_data)

        # convert to jax
        if self.tensor_type == "jax" and not jax.config.jax_enable_x64:
            # explicitly convert in onp.float32 and int32
            # berfore jax conversion to avoid jax warnings.
            def np_convert(x):
                if x.dtype == onp.float64:
                    return x.astype(onp.float32)
                if x.dtype == onp.int64:
                    return x.astype(onp.int32)
                return x

            np_data, np_index, xs = tree_map(np_convert, (np_data, np_index, xs))
        np_data, np_index, xs = ep.convert_to_tensors(
            (np_data, np_index, xs), self.tensor_type
        )

        from wax.compile import jit_init_apply

        @jit_init_apply
        @transform_with_state
        def transform_dataset(step):
            dataset = partial(tree_access_data, np_data, np_index)(step)
            return module(dataset)

        seq = hk.PRNGSequence(42)
        # outputs, state = static_unroll(access_dataset, xs, next(seq))
        outputs, state = dynamic_unroll(transform_dataset, xs, next(seq))

        if self.format_outputs:
            # outputs = ep.convert_to_tensors(outputs, "numpy")
            # TODO: fix eagerpy convert_to_tensor for numpy conversion.
            if isinstance(outputs, jnp.ndarray):
                # try to avoid tree_map which seems a bit slow.
                outputs = onp.array(outputs)
            else:
                outputs = tree_map(lambda x: onp.array(x), outputs)

        if self.return_state:
            return outputs, state
        else:
            return outputs

    def get_master_time(self, time_datasets):
        if not self.master_time:
            if not len(time_datasets) == 1:
                raise ValueError(
                    "master_time must be specified since multiple time index have been detected.",
                    f"It should be in {time_datasets.keys()}",
                )
            master_time = next(iter(time_datasets.keys()))
        else:
            master_time = self.master_time
        return master_time

    def trace_dataset(self, dataset: xr.Dataset) -> Any:
        """Trace dataset time indices in order to syncrhonize them an prepare data access
        through unroll operations.
            Args:
                dataset : dataset on which the transformation is applied

            Return:
                Unroll results of the module formated as a nested data
                structure with dataarray leaves.
        """
        time_dataset = get_time_dataset(dataset)
        time_datasets = split_dataset_from_time_dims(time_dataset)
        master_time = self.get_master_time(time_datasets)

        if len(time_datasets) > 1:
            streams = self.start_dataset_streams(time_datasets)
            stream_merged = self.merge(streams)
            time_index = unroll_stream(stream_merged, pbar=self.pbar)
            # now convert in index for the original dataset.
        else:
            times = time_datasets[master_time][master_time].values
            index = dataset_to_numpy(time_datasets[master_time])
            time_index = {master_time: StreamObservation(times, index)}
        time_dataset_index = get_dataset_index_from_stream_index(time_index)
        dataset_index = get_dataset_index(dataset, time_dataset_index)
        # convert to dict of numpy
        np_data = dataset_to_numpy(dataset)
        np_index = dataset_to_numpy(dataset_index)

        # prepare steps
        xs = onp.arange(len(time_dataset_index[master_time]))
        return np_data, np_index, xs

    def merge(
        self, streams: Dict[str, Generator[None, None, None]]
    ) -> Generator[None, None, None]:
        """Merge multiple streams in one stream_data.

        Args:
            streams: mapping of streams (generators)

        """
        buffer_dtype_init_values = self.buffer_dtype_init_values or DTYPE_INIT_VALUES
        # initialization
        _output_values: Dict = {}
        stream_obs: Dict = {}
        buffers: Dict = {}
        for time_dim, stream in streams.items():
            # initialization of the generators (first call to next)
            stream_obs[time_dim] = next(stream)

            def get_fill_value(x):
                dtype_ = get_unique_dtype(x)
                return buffer_dtype_init_values[dtype_]

            fill_value = StreamObservation(
                pd.NaT,  # time are initialized to None as there is not nan for datetimes in numpy.
                tree_map(get_fill_value, stream_obs[time_dim].value),
            )

            # init obs
            init_obs = StreamObservation(
                pd.NaT,  # time are initialized to None as there is not nan for datetimes in numpy.
                tree_multimap(
                    lambda x, fv: onp.full_like(x, fv),
                    stream_obs[time_dim].value,
                    fill_value.value,
                ),
            )

            # fire the buffers
            if time_dim in self.buffer_maxlen:
                buffers[time_dim] = tree_buffer_transform(
                    self.buffer_maxlen[time_dim], fill_value, init_obs
                )
            else:
                # identity transformation
                buffers[time_dim] = lambda x: x

            buffer_output = buffers[time_dim](init_obs)
            _output_values[time_dim] = buffer_output

        generator_state = GeneratorState(_output_values, stream_obs)

        # initialization
        # yield _output_values

        original_streams = streams.copy()
        master_time = self.get_master_time(streams)
        while True:
            # main loop
            streams = original_streams.copy()

            output_values, stream_obs = generator_state

            main_stream = streams.pop(master_time)

            # initialize outputs with initial values
            output = output_values.copy()
            try:
                main_obs = cast(StreamObservation, next(main_stream))
            except StopIteration:
                return

            output[master_time] = main_obs
            main_time = main_obs.time

            if _is_verbose(self.verbose, master_time):
                print(f"[MultiStream] '{master_time}' proceed data : {main_time}")

            for time_dim, stream in streams.items():
                stream_time = stream_obs[time_dim].time

                if time_dim in self.freqs:
                    embed = cast(
                        Callable[[Any], Any],
                        partial(floor_datetime, freq=self.freqs[time_dim]),
                    )
                else:

                    def embed(x):
                        return x

                if time_less(stream_time, main_time, embed):
                    # put last readed observation
                    if _is_verbose(self.verbose, time_dim):
                        print(f"'{time_dim}' proceed data : {stream_time}")
                    buffer_output = buffers[time_dim](stream_obs[time_dim])
                    output[time_dim] = buffer_output

                while time_less(stream_time, main_time, embed):
                    try:
                        stream_obs[time_dim] = next(stream)
                        stream_time = stream_obs[time_dim].time
                    except StopIteration:
                        del stream_obs[time_dim]

                        # We will not use anymore this stream_data
                        del original_streams[time_dim]
                        break

                    if time_less(stream_time, main_time, embed):
                        buffer_output = buffers[time_dim](stream_obs[time_dim])
                        output[time_dim] = buffer_output
                        if _is_verbose(self.verbose, time_dim):
                            print(f"'{time_dim}' proceed data : {stream_time}")

                if _is_verbose(self.verbose, time_dim):

                    def as_list(vals):

                        vals = onp.array(vals)
                        if vals.ndim == 2:
                            idx = onp.stack([vals[i] for i in range(len(vals))])
                            return pd.DataFrame(output[time_dim].value, index=idx)
                        idx = vals
                        return (output[time_dim].value, idx)

                    print(
                        f"'{time_dim}'  return output: {as_list(output[time_dim].time)}"
                    )

                if self.ffills.get(time_dim, False):
                    # forward fill observation
                    output_values[time_dim] = output[time_dim]

            yield output

    def start_dataset_stream(
        self,
        dataset: xr.Dataset,
        time_dim: Optional[str] = None,
    ):
        """
        Args:
            dataset : xarray Dataset
            time_dim : dimension on witch to iterate
        """
        if time_dim is None:
            time_dim = _get_unique_time_dim(dataset)

        times = dataset.coords[time_dim].values

        # convert to numpy dataset
        np_dataset = dataset_to_numpy(dataset)
        del dataset
        # initialize generator
        i = 0
        time = times[0]
        obs = StreamObservation(time, np_select_item(np_dataset, 0, self.trace))
        if _is_verbose(self.verbose, time_dim):
            print(
                " " * 50 + f"[stream_dataset_generator] Initialization, "
                f"read data '{time_dim}' : index({i})  : '{obs.time}'"
            )
        yield obs

        times = _add_pbar(times, self.pbar, time_dim)
        for i, time in enumerate(times):
            obs = StreamObservation(time, np_select_item(np_dataset, i, self.trace))
            if _is_verbose(self.verbose, time_dim):
                print(
                    " " * 50 + f"[stream_dataset_generator] "
                    f"Read data '{time_dim}' : index({i})  : '{obs.time}'"
                )
            yield obs

    def start_dataset_streams(
        self,
        datasets: Dict[str, xr.Dataset],
    ) -> Dict[str, Generator[None, None, None]]:
        """
        Args:
            datasets: mapping of datasets.
            trace : if True, perform data tracing returning only indexes instead of actual data.

        Returns:
            Dict of generators.
        """
        streams = {
            time_dim: self.start_dataset_stream(tds, time_dim=time_dim)
            for time_dim, tds in datasets.items()
        }
        return streams


def _add_pbar(times, pbar, time_dim):
    if isinstance(pbar, bool):
        if pbar:
            return tqdm(times, desc=time_dim)

    if isinstance(pbar, (list, tuple)):
        if time_dim in pbar:
            return tqdm(times, desc=time_dim)
    return times
