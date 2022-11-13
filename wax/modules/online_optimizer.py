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
"""Online optimizer module."""
from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Union

import haiku as hk
import jax
import jax.numpy as jnp
from haiku.data_structures import partition
from jax.tree_util import tree_map
from optax import GradientTransformation

from wax.modules.optax_optimizer import OptaxOptimizer
from wax.predicate import pass_all_predicate


class ParamsState(NamedTuple):
    trainable_params: Any
    non_trainable_params: Any
    state: Any


@dataclass
class OptInfo:
    """Dynamically define OptimizerInfo namedtuple structure.

    Args:
        return_params: if true, add 'params' field to output structure.
    """

    return_params: bool = False

    def __post_init__(self):
        outputs = ["loss", "model_info"]
        if self.return_params:
            outputs += ["params"]
        self.opt_info_struct_ = namedtuple("OptInfo", outputs)

    def __call__(
        self,
        loss: float,
        model_info: Any,
        params: Any = None,
    ):
        outputs = [loss, model_info]
        if self.return_params:
            outputs += [params]
        return self.opt_info_struct_(*outputs)


class OnlineOptimizer(hk.Module):
    """Wraps a model with loss and and optimizer to perform one online learning update."""

    def __init__(
        self,
        model: Union[Callable, hk.TransformedWithState],
        opt: Union[OptaxOptimizer, GradientTransformation],
        project_params: Optional[Callable] = None,
        regularize_loss: Optional[Callable] = None,
        params_predicate: Callable[[str, str, jnp.ndarray], bool] = pass_all_predicate,
        return_params=False,
        name: Optional[str] = None,
    ):
        """Initialize module.

        Args:
            model : model to optimize. The model should return a tuple (loss, info)
            opt : optimizer: Optax transformation consisting of a function pair: (initialise, update).
            project_params : function to project parameters. It applies to parameters and optimizer state .
            regularize_loss: function to regularize the model loss. It applies to the parameters.
            params_predicate: function to split params in trainable and non-trainable params.
                See https://dm-haiku.readthedocs.io/en/latest/notebooks/non_trainable.html
            name : name of the module
        """
        super().__init__(name=name)
        self.model = (
            model
            if isinstance(model, hk.TransformedWithState)
            else hk.transform_with_state(model)
        )
        self.opt = (
            OptaxOptimizer(opt) if isinstance(opt, GradientTransformation) else opt
        )
        self.project_params = project_params
        self.regularize_loss = regularize_loss
        self.params_predicate = params_predicate

        self.OptInfo = OptInfo(return_params)

    def __call__(self, *args, **kwargs):
        """Update learner.

        Args:
            x : features
            y: target
        """

        def init_model_params_and_state(shape, dtype):
            """Set state from trainable params and state of the model."""
            params, state = self.model.init(hk.next_rng_key(), *args, **kwargs)
            trainable_params, non_trainable_params = partition(
                self.params_predicate, params
            )
            trainable_params = hk.data_structures.to_mutable_dict(trainable_params)
            return ParamsState(trainable_params, non_trainable_params, state)

        trainable_params, non_trainable_params, state = hk.get_state(
            "model_params_and_state",
            [],
            init=init_model_params_and_state,
        )

        step = hk.get_state("step", [], init=lambda *_: 0)

        def _loss(trainable_params, non_trainable_params, state, *args, **kwargs):
            params = hk.data_structures.merge(trainable_params, non_trainable_params)
            (loss, model_info), state = self.model.apply(
                params, state, None, *args, **kwargs
            )
            if self.regularize_loss:
                loss += self.regularize_loss(trainable_params)
            return loss, (model_info, params, state)

        # compute loss and gradients
        (loss, (model_info, params, state)), grads = jax.value_and_grad(
            _loss, has_aux=True
        )(trainable_params, non_trainable_params, state, *args, **kwargs)

        # update optimizer state
        filled_grads = tree_map(jnp.nan_to_num, grads)

        # update params
        updated_trainable_params = self.opt(trainable_params, filled_grads)

        if self.project_params:
            updated_trainable_params = self.project_params(
                updated_trainable_params, self.opt
            )

        updated_params = hk.data_structures.merge(
            updated_trainable_params, non_trainable_params
        )
        step += 1
        hk.set_state("step", step)
        hk.set_state(
            "model_params_and_state",
            ParamsState(updated_trainable_params, non_trainable_params, state),
        )

        opt_info = self.OptInfo(
            loss,
            model_info,
            updated_params,
        )
        return opt_info
