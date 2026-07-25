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
"""Flax-based MaskNormalize module for normalizing data with masked standard deviation."""

from typing import Any

import flax.linen as nn
import jax.numpy as jnp

from .fill_nan_inf import FillNanInf
from .mask_std import MaskStd


class MaskNormalize(nn.Module):
    """Flax-based module for normalizing data by standard deviation computed with mask.

    This module normalizes input data by dividing by the standard deviation
    computed only over the masked (True) regions. Handles edge cases with
    NaN and infinity values.
    """

    axis: int | None = None
    assume_centered: bool = False

    def setup(self):
        """Setup the MaskNormalize module."""
        self.mask_std = MaskStd(axis=self.axis, assume_centered=self.assume_centered)
        self.fill_nan_inf = FillNanInf()

    def __call__(self, mask: jnp.ndarray, input: Any) -> Any:
        """Normalize data by masked standard deviation.

        Args:
            mask: Boolean mask array
            input: Input data to normalize

        Returns:
            Normalized data (input / masked_std) with NaN/Inf handled
        """

        def normalize_fn(x):
            # Compute standard deviation over masked regions
            x_std = self.mask_std(mask, x)

            # Normalize by dividing by standard deviation
            x_normalized = x / x_std

            # Handle NaN and infinity values that may result from division
            x_normalized = self.fill_nan_inf(x_normalized)

            return x_normalized

        # Handle tree structures (nested data)
        from jax.tree_util import tree_map

        return tree_map(normalize_fn, input)


def create_mask_normalize(axis: int | None = None, assume_centered: bool = False) -> MaskNormalize:
    """Factory function to create MaskNormalize module.

    Args:
        axis: Axis for normalization (default: None)
        assume_centered: Whether to assume zero mean (default: False)

    Returns:
        MaskNormalize module instance
    """
    return MaskNormalize(axis=axis, assume_centered=assume_centered)
