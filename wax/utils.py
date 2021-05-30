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
"""Some utils functions used in WAX-ML."""

from jax import tree_flatten


def dict_map(fun, col):
    return {key: fun(val) for key, val in col.items()}


def get_unique_dtype(current_values_):
    # check of unique dtype
    # TODO remove onece multi-dtype is supported
    current_values_flat_, _ = tree_flatten(current_values_)
    current_dtypes_ = set(map(lambda x: x.dtype.type, current_values_flat_))
    assert (
        len(current_dtypes_) == 1
    ), "multi-dtype not yet supported. TODO: manage multi-dtypes at Buffer level."
    current_dtype_ = current_dtypes_.pop()
    return current_dtype_
