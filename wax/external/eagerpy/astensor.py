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
import functools
from numbers import Number
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from jax import tree_flatten, tree_unflatten

from .tensor import (
    JAXTensor,
    NumPyTensor,
    PyTorchTensor,
    Tensor,
    TensorFlowTensor,
    TensorType,
)
from .types import NativeTensor

if TYPE_CHECKING:
    # for static analyzers
    import torch


def _get_module_name(x: Any) -> str:
    # splitting is necessary for TensorFlow tensors
    return type(x).__module__.split(".")[0]


@overload
def astensor(x: TensorType) -> TensorType:
    ...


@overload
def astensor(x: "torch.Tensor") -> PyTorchTensor:
    ...


@overload
def astensor(x: NativeTensor) -> Tensor:  # type: ignore
    ...


def astensor(x: Union[NativeTensor, Tensor, Any]) -> Union[Tensor, Any]:  # type: ignore
    # we use the module name instead of isinstance
    # to avoid importing all the frameworks
    name = _get_module_name(x)

    if name == "torch":
        return PyTorchTensor(x)  # type: ignore
    if name == "tensorflow":
        return TensorFlowTensor(x)
    if name == "jax" or name == "jaxlib":
        return JAXTensor(x)  # type: ignore
    if name == "numpy":
        return NumPyTensor(x)
    if isinstance(x, (str, Number)):
        return NumPyTensor(x)

    # non Tensor types are returned unmodified
    return x


def astensors(data: Any, *args) -> Any:  # type: ignore
    if args:
        data = (data,) + args
    leaf_values, tree_def = tree_flatten(data)
    leaf_values = tuple(astensor(value) for value in leaf_values)
    return tree_unflatten(tree_def, leaf_values)


T = TypeVar("T")


class RestoreTypeFunc(Generic[T]):
    def __init__(self, x: T):
        self.unwrap = not isinstance(x, Tensor)

    @overload
    def __call__(self, x: Tensor) -> T:
        ...

    @overload  # noqa: F811
    def __call__(self, x: Tensor, y: Tensor) -> Tuple[T, T]:
        ...

    @overload  # noqa: F811
    def __call__(self, x: Tensor, y: Tensor, z: Tensor, *args: Tensor) -> Tuple[T, ...]:
        ...

    @overload  # noqa: F811
    def __call__(self, *args: Any) -> Any:
        # catch other types, otherwise we would return type T for input type Any
        ...

    def __call__(self, *tensors):  # type: ignore  # noqa: F811
        result = tuple(self._restore(x) for x in tensors) if self.unwrap else tensors
        if len(result) == 1:
            (result,) = result
        return result

    def _restore(self, x):  # type: ignore  # noqa: F811
        x_raw = as_raw_tensors(x)
        if isinstance(x_raw, tuple):
            assert len(x_raw) == 2
            x_raw, unwrap = x_raw
            assert unwrap
            return x_raw
        else:
            return x_raw


def astensor_(x: T) -> Tuple[Tensor, RestoreTypeFunc[T]]:
    return astensor(x), RestoreTypeFunc[T](x)


def astensors_(x: T, *xs: T) -> Tuple[Tuple[Tensor, ...], RestoreTypeFunc[T]]:
    return astensors(x, *xs), RestoreTypeFunc[T](x)


def has_tensor(tree_def: Any) -> bool:
    if "NumPyTensor" in str(tree_def):
        return True
    if "JAXTensor" in str(tree_def):
        return True
    if "TensorFlowTensor" in str(tree_def):
        return True
    if "PyTorchTensor" in str(tree_def):
        return True
    return False


def as_tensors_any(data: Any) -> Tuple[Any, bool]:
    """Convert data structure leaves in Tensor and detect if any of the input data contains a Tensor.

    Parameters
    ----------
    data
        data structure.

    Returns
    -------
    Any
        modified data structure.
    bool
        True if input data contains a Tensor type.
    """
    leaf_values, tree_def = tree_flatten(data)
    transformed_leaf_values = tuple(astensor(value) for value in leaf_values)
    return tree_unflatten(tree_def, transformed_leaf_values), has_tensor(tree_def)


def _is_tensor(x: T) -> bool:
    name = _get_module_name(x)

    if name == "torch":
        return True
    if name == "tensorflow":
        return True
    if name == "jax" or name == "jaxlib":
        return True
    if name == "numpy":
        return True
    if isinstance(x, (str, int)):
        return True
    return False


def as_raw_tensor_leave(x: T) -> Any:
    if _is_tensor(x):
        unwrap = True
        return (x, unwrap)
    else:
        return x


def as_raw_tensor(x: T) -> Any:
    if isinstance(x, Tensor):
        unwrap = True
        return (x.raw, unwrap)
    else:
        return x


def as_raw_tensors(data: Any) -> Any:
    leaf_values, tree_def = tree_flatten(data)
    if not has_tensor(tree_def):
        return data
    leaf_values = tuple(as_raw_tensor(value) for value in leaf_values)
    leaf_values_raw = list(map(as_raw_tensor_leave, leaf_values))
    data_raw = tree_unflatten(tree_def, leaf_values_raw)
    return data_raw


def eager_function(func: Callable[..., T]) -> Callable[..., T]:
    @functools.wraps(func)
    def eager_func(*args: Any, **kwargs: Any) -> Any:
        (args, kwargs), has_tensor = as_tensors_any((args, kwargs))
        unwrap = not has_tensor
        result = func(*args, **kwargs)
        if unwrap:
            raw_result = as_raw_tensors(result)
            return raw_result
        else:
            return result

    return eager_func
