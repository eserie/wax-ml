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
"""Compute standard deviation  over masked data."""

import haiku as hk
import jax.numpy as jnp
from jax import tree_map

from wax.modules.apply_mask import ApplyMask


class MaskStd(hk.Module):
    """Compute standard deviation  over masked data."""

    def __init__(self, axis=None, assume_centered=False, name=None):
        """Initialize the module.

        Args:
            axis : axis along which to compute the mean
            assume_centered : if True assume mean to be zero when computing
                the standard deviation.
        """
        super().__init__(name=name)
        self.axis = axis
        self.assume_centered = assume_centered

    def __call__(self, mask, input):
        """Compute standard deviation.

        Args:
            mask : bolean mask specifying points over which to compute the mean.
            input : data on which to compute the mean.
        """

        def mask_std(x):
            # mask values
            x = ApplyMask()(mask, x)

            # fill nans
            x = jnp.nan_to_num(x)

            count = mask.sum(axis=self.axis)
            mean = x.sum(axis=self.axis) / count

            if self.assume_centered:
                diff = x
            else:
                diff = x - mean

            diff = ApplyMask()(mask, diff)
            std = jnp.sqrt((diff ** 2).sum(axis=self.axis) / count)
            return std

        return tree_map(mask_std, input)
