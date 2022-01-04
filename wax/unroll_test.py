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
import haiku as hk
import jax
import jax.numpy as jnp

from wax.unroll import unroll


class GenModel(hk.Module):
    def __call__(self, i):
        return hk.next_rng_key()[0]


def test_unroll_rng():
    fun = hk.transform(lambda x: (GenModel()(x), GenModel()(x)))
    rng = jax.random.PRNGKey(42)
    params = fun.init(rng, 1)
    ref = []
    for i in range(5):
        (rng, sub_rng) = jax.random.split(rng)
        ref.append(fun.apply(params, sub_rng, 1))
    ref = jnp.array(ref)

    rng = jax.random.PRNGKey(42)
    out = unroll(lambda x: (GenModel()(x), GenModel()(x)), rng=rng, dynamic=False)(
        jnp.arange(5)
    )
    assert (ref == jnp.stack(out).T).flatten().all()
