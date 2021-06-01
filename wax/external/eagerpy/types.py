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
from typing import TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    # for static analyzers
    import jax  # noqa: F401
    import numpy  # noqa: F401
    import tensorflow  # noqa: F401
    import torch  # noqa: F401

Axes = Tuple[int, ...]
AxisAxes = Union[int, Axes]
Shape = Tuple[int, ...]
ShapeOrScalar = Union[Shape, int]

# tensorflow.Tensor, jax.numpy.ndarray and numpy.ndarray currently evaluate to Any
# we can therefore only provide additional type information for torch.Tensor
NativeTensor = Union[
    "torch.Tensor", "tensorflow.Tensor", "jax.numpy.ndarray", "numpy.ndarray"
]
