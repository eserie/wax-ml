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
"""The online newton optimizer. It extends optax optimizers."""
from typing import NamedTuple

import jax
import jax.numpy as jnp
from optax._src import base, combine
from optax._src.alias import ScalarOrSchedule, _scale_by_learning_rate


def newton(
    learning_rate: ScalarOrSchedule, eps: float = 1e-5
) -> base.GradientTransformation:
    """The online newton optimizer.

    It extends optax optimizers.

    References
    ----------
    [^1] [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for
    online convex optimization. Machine Learning, 69(2-3), pp.169-192]
    (https://link.springer.com/content/pdf/10.1007/s10994-007-5016-8.pdf)

    Args:
      learning_rate: this is a fixed global scaling factor.
      initial_accumulator_value: initialisation for the accumulator.
      eps: a small constant applied to denominator inside of the square root
        (as in RMSProp) to avoid dividing by zero when rescaling.

    Returns:
      the corresponding `GradientTransformation`.
    """
    return combine.chain(
        scale_by_newton(eps=eps),
        _scale_by_learning_rate(learning_rate),
    )


class ScaleByNewtonState(NamedTuple):
    """State holding the sum of gradient squares to date."""

    hessian_inv: base.Updates


def sherman_morrison(A_inv, u, v):
    den = 1.0 + (u @ A_inv) @ v
    A_inv -= A_inv @ jnp.outer(u, v) @ A_inv / den
    return A_inv


def scale_by_newton(eps: float = 1e-7) -> base.GradientTransformation:
    """Rescale updates by multiplying by the inverse of an approximation of the hessian.

    References
    ----------
    [^1]: [Hazan, E., Agarwal, A. and Kale, S., 2007. Logarithmic regret algorithms for online convex optimization. Machine Learning, 69(2-3), pp.169-192](https://www.cs.princeton.edu/~ehazan/papers/log-journal.pdf)

    Args:
      eps: A small floating point value to avoid zero denominator.

    Returns:
      An (init_fn, update_fn) tuple.
    """

    def init_fn(params):
        hessian_inv = jax.tree_util.tree_map(
            lambda t: jnp.eye(len(t.flatten()), dtype=t.dtype) / eps, params
        )
        return ScaleByNewtonState(hessian_inv=hessian_inv)

    def update_fn(updates, state, params=None):
        del params

        class Tuple(tuple):
            """Class to avoid pytree conversion and allow for the use
            of shapes in final reshape."""

            ...

        shapes = jax.tree_util.tree_map(lambda x: Tuple(x.shape), updates)
        updates = jax.tree_util.tree_map(lambda x: x.flatten(), updates)
        hessian_inv = jax.tree_util.tree_map(
            lambda u, hinv: sherman_morrison(hinv, u, u), updates, state.hessian_inv
        )
        updates = jax.tree_util.tree_map(lambda hinv, g: hinv @ g, hessian_inv, updates)
        updates = jax.tree_util.tree_map(
            lambda u, shape: u.reshape(shape), updates, shapes
        )

        return updates, ScaleByNewtonState(hessian_inv=hessian_inv)

    return base.GradientTransformation(init_fn, update_fn)
