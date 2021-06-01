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
from typing import Optional, Union

from .framework import inf
from .tensor import TensorType
from .types import AxisAxes


def l0(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return (x != 0).sum(axis=axis, keepdims=keepdims).astype(x.dtype)


def l1(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return x.abs().sum(axis=axis, keepdims=keepdims)


def l2(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return x.square().sum(axis=axis, keepdims=keepdims).sqrt()


def linf(
    x: TensorType, axis: Optional[AxisAxes] = None, keepdims: bool = False
) -> TensorType:
    return x.abs().max(axis=axis, keepdims=keepdims)


def lp(
    x: TensorType,
    p: Union[int, float],
    axis: Optional[AxisAxes] = None,
    keepdims: bool = False,
) -> TensorType:
    if p == 0:
        return l0(x, axis=axis, keepdims=keepdims)
    if p == 1:
        return l1(x, axis=axis, keepdims=keepdims)
    if p == 2:
        return l2(x, axis=axis, keepdims=keepdims)
    if p == inf:
        return linf(x, axis=axis, keepdims=keepdims)
    return x.abs().pow(p).sum(axis=axis, keepdims=keepdims).pow(1.0 / p)
