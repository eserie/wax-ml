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
"""Normalize data by its standard deviation computed with a mask."""
import haiku as hk
from jax import tree_map

from wax.modules.fill_nan_inf import FillNanInf
from wax.modules.mask_std import MaskStd


class MaskNormalize(hk.Module):
    """Normalize data by its standard deviation computed with a mask."""

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
        """Normalize data.

        Args:
            mask : bolean mask specifying points over which to compute the mean.
            input : data on which to compute the mean.
        """

        def normalize(x):
            x_std = MaskStd(axis=self.axis, assume_centered=self.assume_centered)(
                mask, x
            )
            x_normalized = FillNanInf()(x / x_std)
            return x_normalized

        return tree_map(normalize, input)
