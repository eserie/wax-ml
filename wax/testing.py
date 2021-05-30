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
"""Defines a comparison function to compare two nested data structures."""
from numbers import Number

from jax import numpy as onp
from jax import tree_flatten
from jax._src.numpy.lax_numpy import int32, uint32


def assert_tree_all_close(x, y, check_treedef=True):
    from jax.config import config

    config.update("jax_enable_x64", False)

    x, x_treedef = tree_flatten(x)
    y, y_treedef = tree_flatten(y)
    if check_treedef:
        ref_str = str(x_treedef).replace("wax.", "")
        check_str = str(y_treedef).replace("_test", "").replace("wax.", "")
        assert ref_str == check_str, f"{ref_str} != {check_str}"
    assert len(x) == len(y)
    for x_, y_ in zip(x, y):
        assert type(x_) == type(y_)

        if isinstance(x_, Number):
            assert x_ == y_
        else:
            assert x_.dtype == y_.dtype, f"{x_.dtype} != {y_.dtype}"
            if x_.dtype in [int32, uint32]:
                assert (x_ == y_).all()
            else:
                assert onp.allclose(x_, y_)
