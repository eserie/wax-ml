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
from typing import Any, Callable, NamedTuple, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import optax

from wax.modules import FillNanInf


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
        model: Union[Callable, hk.TransformedWithState],
        opt: Any,
        loss: Callable,
        grads_fill_nan_inf=True,
        name: str = None,
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
        self.model = (
            model
            if isinstance(model, hk.TransformedWithState)
            else hk.transform_with_state(model)
        )
        self.opt = opt
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

        step = hk.get_state("step", [], init=lambda *_: jnp.array(0))
        params, state = hk.get_state(
            "model_params_state",
            [],
            init=lambda *_: ParamsState(*self.model.init(hk.next_rng_key(), x)),  # type: ignore
        )
        opt_state = hk.get_state("opt_state", [], init=lambda *_: self.opt.init(params))

        @jax.jit
        def _loss(params, state, x, y):
            y_pred, state = self.model.apply(params, state, None, x)
            return self.loss(y_pred, y), (y_pred, state)

        # compute loss and gradients
        (l, (y_pred, state)), grads = jax.value_and_grad(_loss, has_aux=True)(
            params, state, x, y
        )

        if self.grads_fill_nan_inf:
            grads = FillNanInf()(grads)

        # update optimizer state
        grads, opt_state = self.opt.update(grads, opt_state)

        # update params
        params = optax.apply_updates(params, grads)

        step += 1
        hk.set_state("step", step)
        hk.set_state("model_params_state", ParamsState(params, state))
        hk.set_state("opt_state", opt_state)

        info = OnlineSupervisedLearnerInfo(loss=l, params=params)
        return y_pred, info
