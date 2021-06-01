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
from functools import partial
from typing import Any

from jax.tree_util import tree_flatten, tree_unflatten

from .astensor import _get_module_name


def convert_to_tensor(data: Any, tensor_type: str) -> Any:
    """Convert a tensor in a given tensor_type.

    Parameters
    ----------
    tensor_type
        The targeted tensor type. Can be in ['numpy', 'tensorflow', 'jax', 'torch"].

    Returns
    -------
    data
        data structure with converted tensors.
    """
    name = _get_module_name(data)
    if name not in ["numpy", "jax", "torch", "tensorflow"]:
        # do not convert
        return data

    if tensor_type == "tensorflow":
        import tensorflow as tf

        return tf.convert_to_tensor(data)

    elif tensor_type == "torch":
        import torch

        return torch.tensor(data)

    elif tensor_type == "jax":
        import jax.numpy as jnp

        return jnp.asarray(data, dtype=data.dtype)

    elif tensor_type == "numpy":
        import numpy as onp

        return onp.asarray(data)

    raise ValueError(
        f"tensor_type {tensor_type} must be in ['numpy', 'tensorflow', 'jax', 'torch']"
    )


def convert_to_tensors(data: Any, tensor_type: str) -> Any:
    """Convert tensors in a nested data structure .

    Parameters
    ----------
    tensor_type
        The targeted tensor type. Can be in ['numpy', 'tensorflow', 'jax', 'torch"].

    Returns
    -------
    data
        data structure with converted tensors.
    """
    leaf_values, tree_def = tree_flatten(data)
    leaf_values = list(
        map(partial(convert_to_tensor, tensor_type=tensor_type), leaf_values)
    )
    return tree_unflatten(tree_def, leaf_values)
