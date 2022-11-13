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
"""Online supervised learner."""
from typing import Any, Callable, NamedTuple, Optional, Tuple, Union

import haiku as hk
import jax.numpy as jnp
from optax import GradientTransformation

from wax.modules.func_optimizer import FuncOptimizer
from wax.modules.optax_optimizer import OptaxOptimizer


class ParamsState(NamedTuple):
    params: Any
    state: Any


class OnlineSupervisedLearnerInfo(NamedTuple):
    loss: float
    params: Any


class OnlineSupervisedLearner(hk.Module):
    """Online supervised learner."""

    def __init__(
        self,
        model: Callable,
        opt: Union[OptaxOptimizer, GradientTransformation],
        loss: Callable,
        grads_fill_nan_inf=True,
        name: Optional[str] = None,
    ):
        """Initialize module.

        Args:
            model : model to optimize.
            opt : optimizer.
            loss : loss function.
            name : name of the module
            grads_fill_nan_inf: if true, fill nan and
                +/- infinite values in gradients with zeros.
        """
        super().__init__(name=name)
        self.model = model
        self.opt = (
            OptaxOptimizer(opt) if isinstance(opt, GradientTransformation) else opt
        )
        self.loss = loss
        self.grads_fill_nan_inf = grads_fill_nan_inf

    def __call__(
        self, x: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[jnp.ndarray, OnlineSupervisedLearnerInfo]:
        """Update learner.

        Args:
            x : features
            y: target
        """

        def _loss(x, y):
            y_pred = self.model(x)
            return self.loss(y, y_pred), y_pred

        (l, y_pred), params = FuncOptimizer(
            _loss, self.opt, has_aux=True, grads_fill_nan_inf=self.grads_fill_nan_inf
        )(x, y)
        info = OnlineSupervisedLearnerInfo(loss=l, params=params)
        return y_pred, info
