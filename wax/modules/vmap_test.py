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
import jax
import jax.numpy as jnp

from wax.modules import EWMA, VMap
from wax.stateful import unroll_lift_with_state
from wax.unroll import unroll


def test_vmap_with_outer_unroll_lift_with_state():
    """Test that VMap module work without PRNG key specified"""
    x = jnp.arange(2 * 2 * 2).reshape(2, 2, 2).astype(jnp.float32)

    def outer_fun(x):
        @unroll_lift_with_state
        def step(x):
            return VMap(EWMA(com=10))(x)

        return step(x)

    res = unroll(outer_fun, rng=jax.random.PRNGKey(0))(x)
    assert res.shape == x.shape
    ref = jnp.array(
        [
            [[0.0, 1.0], [1.0476191, 2.047619]],
            [[2.1268883, 3.1268883], [3.2376645, 4.2376647]],
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(res, ref)
