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
"""Flax-based ApplyMask module for masking data arrays."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp


class ApplyMask(nn.Module):
    """Flax-based module for applying boolean masks to data arrays.

    This module applies boolean masks to input data, setting masked-out values
    to a specified mask_value. Supports axis-specific operations.
    """

    axis: int | None = None
    mask_value: float = 0.0

    def __call__(self, mask: jnp.ndarray, input: Any) -> Any:
        """Apply mask to input data.

        Args:
            mask: Boolean mask array
            input: Input data to be masked

        Returns:
            Masked data with False mask regions set to mask_value

        Raises:
            ValueError: If axis > 1 (not supported)
        """
        if self.axis is not None and self.axis > 1:
            raise ValueError(f"axis={self.axis} > 1 not supported.")

        def apply_mask_fn(x):
            # Validate input is array-like
            if not isinstance(x, jnp.ndarray):
                x = jnp.asarray(x)

            # Handle axis-specific broadcasting
            if self.axis is not None:
                # Ensure mask can broadcast with data along specified axis
                if self.axis == 0:
                    # Mask broadcasts along axis 0 - expand dimensions as needed
                    mask_broadcast = mask
                    while mask_broadcast.ndim < x.ndim:
                        mask_broadcast = mask_broadcast[..., None]
                elif self.axis == 1:
                    # Mask broadcasts along axis 1 - expand dimensions as needed
                    mask_broadcast = mask
                    if mask_broadcast.ndim == 1 and x.ndim > 1:
                        mask_broadcast = mask_broadcast[None, :]
                    while mask_broadcast.ndim < x.ndim:
                        mask_broadcast = mask_broadcast[..., None]
                else:
                    mask_broadcast = mask
            else:
                mask_broadcast = mask

            # Apply mask: keep x where mask is True, use mask_value where False
            return jnp.where(mask_broadcast, x, self.mask_value)

        # Handle tree structures (nested data)
        from jax.tree_util import tree_map

        return tree_map(apply_mask_fn, input)


def create_apply_mask(axis: int | None = None, mask_value: float = 0.0) -> ApplyMask:
    """Factory function to create ApplyMask module.

    Args:
        axis: Optional axis specification (None, 0, or 1 supported)
        mask_value: Value to use where mask is False

    Returns:
        ApplyMask module instance
    """
    return ApplyMask(axis=axis, mask_value=mask_value)
