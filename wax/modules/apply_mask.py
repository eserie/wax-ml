# Copyright 2021 The Wax Authors
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
from functools import partial

import haiku as hk
import jax.numpy as jnp
from jax import tree_map


class ApplyMask(hk.Module):
    def __init__(self, axis=None, mask_value=0.0, name=None):
        super().__init__(name=name)
        self.axis = axis
        self.mask_value = mask_value

    def __call__(self, mask, input):
        def apply_mask(mask, x):
            # mask values
            if self.axis is not None:
                if self.axis == 0:
                    mask = jnp.repeat(mask.reshape(-1, 1), x.shape[1], axis=1)
                elif self.axis == 1:
                    mask = jnp.repeat(mask.reshape(1, -1), x.shape[0], axis=0)
                else:
                    # TODO: implement for any value of axis
                    raise ValueError(
                        "ApplyMask is not implemented for "
                        "axis different from None, 0, 1."
                    )

            if x.shape != mask.shape:
                raise ValueError("mask shape and data shape are not coherent")
            return jnp.where(mask, x, self.mask_value)

        return tree_map(partial(apply_mask, mask), input)
