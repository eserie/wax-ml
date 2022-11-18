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
import warnings
from typing import Callable, Union

import haiku as hk
import jax
from haiku import TransformedWithState

from wax.stateful import vmap_lift_with_state


class VMap(hk.Module):
    def __init__(self, fun: Callable, name=None):
        """A Haiku module that applies a function to a batch of inputs."""
        warnings.warn(
            "VMap is deprecated, please use vmap_lift_with_state instead.",
            DeprecationWarning,
        )
        super().__init__(name=name)
        self.fun = fun

    def __call__(self, *args, **kwargs):
        return vmap_lift_with_state(self.fun, split_rng=False, init_rng=True)(
            *args, **kwargs
        )


# Helper functions


def add_batch(fun: Union[Callable, TransformedWithState], take_mean=True):
    """Wrap a function with vmap_lift_with_state.
    It should be used inside a transformed function."""

    def fun_batch(*args, **kwargs):
        try:
            res = vmap_lift_with_state(fun, split_rng=True)(*args, **kwargs)
        except ValueError:
            res = vmap_lift_with_state(fun, split_rng=False)(*args, **kwargs)

        if take_mean:
            res = jax.tree_util.tree_map(lambda x: x.mean(axis=0), res)
        return res

    return fun_batch
