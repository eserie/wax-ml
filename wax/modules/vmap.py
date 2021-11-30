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
from typing import Callable, Union

import haiku as hk
import jax
from haiku import TransformedWithState


class VMap(hk.Module):
    def __init__(self, fun: Union[Callable, TransformedWithState], name=None):
        super().__init__(name=name)
        self.fun = (
            fun
            if isinstance(fun, TransformedWithState)
            else hk.transform_with_state(fun)
        )

    def __call__(self, *args, **kwargs):
        n_batches = len(jax.tree_leaves((args, kwargs))[0])
        try:
            rng = hk.next_rng_key()
            rng = jax.random.split(rng, num=n_batches)
        except ValueError:
            rng = None
        params, state = hk.get_state(
            "params_state",
            [],
            init=lambda *_: jax.vmap(self.fun.init)(rng, *args, **kwargs),
        )
        res, state = jax.vmap(self.fun.apply)(params, state, rng, *args, **kwargs)
        hk.set_state("params_state", (params, state))

        return res


# Helper functions


def add_batch(fun: Union[Callable, TransformedWithState], take_mean=True):
    """Wrap a function with VMap module.
    It should be used inside a transformed function."""

    def fun_batch(*args, **kwargs):
        res = VMap(fun)(*args, **kwargs)
        if take_mean:
            res = jax.tree_map(lambda x: x.mean(axis=0), res)
        return res

    return fun_batch
