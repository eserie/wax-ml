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
import jax.numpy as jnp

from wax.modules import VMap
from wax.unroll import unroll


def test_vmap_without_prng_key():
    """Test that VMap module work without PRNG key specified"""
    x = jnp.arange(10).reshape(2, 5)

    def outer_fun(x):
        def fun(x):
            return x

        return VMap(fun)(x)

    unroll(outer_fun)(x)
