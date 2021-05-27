from typing import Any, Optional

import haiku as hk
import jax.numpy as jnp
from jax import tree_map


class FillNanInf(hk.Module):
    """Fill nan, posinf and neginf values.
    Args:
        fill_value : value used to replace nan, posinf or neginf encountered values.
        name : name of the module
    """

    def __init__(self, fill_value: Any = 0.0, name: Optional[str] = None):
        super().__init__(name=name)
        self.fill_value = fill_value

    def __call__(self, input):
        def fill_nan(x):
            # mask values
            return jnp.nan_to_num(
                x, nan=self.fill_value, posinf=self.fill_value, neginf=self.fill_value
            )

        return tree_map(fill_nan, input)
