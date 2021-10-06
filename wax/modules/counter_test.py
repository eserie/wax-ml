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
from haiku._src.data_structures import FlatMapping
from jax import numpy as jnp
from jax._src.numpy.lax_numpy import array as DeviceArray
from jax._src.numpy.lax_numpy import uint32

from wax.modules.counter import Counter
from wax.testing import assert_tree_all_close
from wax.unroll import unroll


def test_counter():
    res, state = unroll(lambda _: Counter()(), return_final_state=True)(
        jnp.arange(10, 20),
    )
    assert_tree_all_close(res, jnp.arange(1, 11, dtype="uint32"))
    assert_tree_all_close(
        state,
        FlatMapping({"counter": FlatMapping({"count": DeviceArray(10, dtype=uint32)})}),
    )
