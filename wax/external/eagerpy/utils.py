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
from typing import overload

from typing_extensions import Literal

from . import modules
from .tensor import JAXTensor, NumPyTensor, PyTorchTensor, Tensor, TensorFlowTensor


@overload
def get_dummy(framework: Literal["pytorch"]) -> PyTorchTensor:
    ...


@overload
def get_dummy(framework: Literal["tensorflow"]) -> TensorFlowTensor:
    ...


@overload
def get_dummy(framework: Literal["jax"]) -> JAXTensor:
    ...


@overload
def get_dummy(framework: Literal["numpy"]) -> NumPyTensor:
    ...


@overload
def get_dummy(framework: str) -> Tensor:
    ...


def get_dummy(framework: str) -> Tensor:
    x: Tensor
    if framework == "pytorch":
        x = modules.torch.zeros(0)
        assert isinstance(x, PyTorchTensor)
    elif framework == "pytorch-gpu":
        x = modules.torch.zeros(0, device="cuda:0")  # pragma: no cover
        assert isinstance(x, PyTorchTensor)  # pragma: no cover
    elif framework == "tensorflow":
        x = modules.tensorflow.zeros(0)
        assert isinstance(x, TensorFlowTensor)
    elif framework == "jax":
        x = modules.jax.numpy.zeros(0)
        assert isinstance(x, JAXTensor)
    elif framework == "numpy":
        x = modules.numpy.zeros(0)
        assert isinstance(x, NumPyTensor)
    else:
        raise ValueError(f"unknown framework: {framework}")  # pragma: no cover
    return x.float32()
