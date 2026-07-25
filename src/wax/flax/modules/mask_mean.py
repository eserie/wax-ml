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
"""Flax-based MaskMean module for computing means over masked data."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from .apply_mask import ApplyMask


class MaskMean(nn.Module):
    """Flax-based module for computing mean over masked data points.

    This module computes the mean of input data considering only the regions
    where the mask is True. Supports axis-specific operations.
    """

    axis: int | None = None

    def setup(self):
        """Setup the MaskMean module."""
        self.apply_mask = ApplyMask(axis=self.axis)

    def __call__(self, mask: jnp.ndarray, input: Any) -> Any:
        """Compute mean over masked data.

        Args:
            mask: Boolean mask array
            input: Input data

        Returns:
            Mean computed over masked regions
        """

        def mask_mean_fn(x):
            # Apply mask (sets False regions to 0)
            x_masked = self.apply_mask(mask, x)

            # Convert NaNs to 0 for proper summation
            x_masked = jnp.nan_to_num(x_masked)

            # Count valid (True) mask entries
            count = mask.sum(axis=self.axis)

            # Avoid division by zero
            count = jnp.where(count == 0, 1, count)

            # Compute mean: sum over valid entries divided by count
            mean = x_masked.sum(axis=self.axis) / count

            # Set mean to NaN where no valid entries exist
            mean = jnp.where(mask.sum(axis=self.axis) == 0, jnp.nan, mean)

            return mean

        # Handle tree structures (nested data)
        from jax.tree_util import tree_map

        return tree_map(mask_mean_fn, input)


def create_mask_mean(axis: int | None = None) -> MaskMean:
    """Factory function to create MaskMean module.

    Args:
        axis: Axis along which to compute the mean (default: None)

    Returns:
        MaskMean module instance
    """
    return MaskMean(axis=axis)
