"""Compute mean over masked data."""

import haiku as hk
import jax.numpy as jnp
from jax import tree_map

from wax.modules.apply_mask import ApplyMask


class MaskMean(hk.Module):
    """Compute mean over masked data."""

    def __init__(self, axis=None, name=None):
        """Initialize the module.
        Args:
            axis : axis along which to compute the mean
        """
        super().__init__(name=name)
        self.axis = axis

    def __call__(self, mask, input):
        """Compute mean.

        Args:
            mask : bolean mask specifying points over which to compute the mean.
            input : data on which to compute the mean.
        """

        def mask_mean(x):
            # mask values
            x = ApplyMask()(mask, x)

            # fill nans
            x = jnp.nan_to_num(x)

            count = mask.sum(axis=self.axis)
            mean = x.sum(axis=self.axis) / count
            return mean

        return tree_map(mask_mean, input)
