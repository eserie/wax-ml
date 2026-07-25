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
"""Flax-based MaskStd module for computing standard deviation over masked data."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from .apply_mask import ApplyMask


class MaskStd(nn.Module):
    """Flax-based module for computing standard deviation over masked data points.

    This module computes the standard deviation of input data considering only
    the regions where the mask is True. Supports axis-specific operations and
    optional centering assumption.
    """

    axis: int | None = None
    assume_centered: bool = False

    def setup(self):
        """Setup the MaskStd module."""
        self.apply_mask = ApplyMask(axis=self.axis)

    def __call__(self, mask: jnp.ndarray, input: Any) -> Any:
        """Compute standard deviation over masked data.

        Args:
            mask: Boolean mask array
            input: Input data

        Returns:
            Standard deviation computed over masked regions
        """

        def mask_std_fn(x):
            # Numerical epsilon for stability
            eps = jnp.finfo(x.dtype).eps

            # Apply mask (sets False regions to 0)
            x_masked = self.apply_mask(mask, x)

            # Convert NaNs to 0 for proper computation
            x_masked = jnp.nan_to_num(x_masked)

            # Count valid (True) mask entries
            count = mask.sum(axis=self.axis)

            # Avoid division by zero
            count = jnp.where(count == 0, 1, count)

            # Compute mean for centering (unless assume_centered=True)
            if self.assume_centered:
                diff = x_masked
            else:
                mean = x_masked.sum(axis=self.axis) / count
                # Broadcast mean back to original shape for subtraction
                if self.axis is not None:
                    if self.axis == 0:
                        mean = mean[None, ...] if x.ndim > 1 else mean
                    elif self.axis == 1:
                        mean = mean[..., None] if x.ndim > 1 else mean
                diff = x - mean

                # Re-mask the differences
                diff = self.apply_mask(mask, diff)

            # Compute variance: sum of squared differences divided by count
            var = (diff**2).sum(axis=self.axis) / count

            # Compute standard deviation with numerical stability
            std = jnp.where(var > 0.0, jnp.sqrt(eps + var), 0.0)

            # Set std to NaN where no valid entries exist
            std = jnp.where(mask.sum(axis=self.axis) == 0, jnp.nan, std)

            return std

        # Handle tree structures (nested data)
        from jax.tree_util import tree_map

        return tree_map(mask_std_fn, input)


def create_mask_std(axis: int | None = None, assume_centered: bool = False) -> MaskStd:
    """Factory function to create MaskStd module.

    Args:
        axis: Axis along which to compute std (default: None)
        assume_centered: If True, assumes mean=0 (default: False)

    Returns:
        MaskStd module instance
    """
    return MaskStd(axis=axis, assume_centered=assume_centered)
