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


class VMap(hk.Module):
    def __init__(self, fun, name=None):
        super().__init__(name=name)
        self.fun = (
            fun
            if isinstance(fun, hk.TransformedWithState)
            else hk.transform_with_state(fun)
        )

    def __call__(self, *args, **kwargs):
        n_batches = len(jax.tree_leaves((args, kwargs))[0])
        rng = hk.next_rng_key()
        rng = jax.random.split(rng, num=n_batches)
        params, state = hk.get_state(
            "params_state",
            [],
            init=lambda *_: jax.vmap(self.fun.init)(rng, *args, **kwargs),
        )
        res, state = jax.vmap(self.fun.apply)(params, state, rng, *args, **kwargs)
        hk.set_state("params_state", (params, state))

        return res
