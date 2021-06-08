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
from typing import Any, Callable, NamedTuple

import haiku as hk
import jax
import optax


class ParamsState(NamedTuple):
    params: Any
    state: Any


class OnlineSupervisedLearner(hk.Module):
    """Online supervised learner."""

    def __init__(self, model: Any, opt: Any, loss: Callable, name: str = None):
        """Initialize module.

        Args:
            model : model to optimize.
            opt : optimizer.
            loss : loss function.
            name : name of the module
        """
        super().__init__(name=name)
        self.model = model
        self.opt = opt
        self.loss = loss

    def __call__(self, x, y):
        """Update learner.

        Args:
            x : features
            y: target
        """

        step = hk.get_state("step", [], init=lambda *_: 0)
        params, state = hk.get_state(
            "model_params_state",
            [],
            init=lambda *_: ParamsState(*self.model.init(hk.next_rng_key(), x)),
        )
        opt_state = hk.get_state("opt_state", [], init=lambda *_: self.opt.init(params))

        @jax.jit
        def _loss(params, state, x, y):
            y_pred, state = self.model.apply(params, state, None, x)
            return self.loss(y_pred, y)

        # compute loss and gradients
        l, grads = jax.value_and_grad(_loss)(params, state, x, y)

        # compute prediction and update model state
        y_pred, state = self.model.apply(params, state, None, x)

        # update optimizer state
        grads, opt_state = self.opt.update(grads, opt_state)

        # update params
        params = optax.apply_updates(params, grads)

        step += 1
        hk.set_state("step", step)
        hk.set_state("model_params_state", ParamsState(params, state))
        hk.set_state("opt_state", opt_state)
        return {
            "loss": l,
            "y_pred": y_pred,
            "params": params,
        }
