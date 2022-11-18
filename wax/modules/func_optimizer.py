# Copyright 2022 The WAX-ML Authors
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
from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp

from wax.modules import FillNanInf
from wax.modules.update_params import UpdateParams, get_init_params
from wax.predicate import pass_all_predicate


class FuncOptimizer(hk.Module):
    """Optimize a function with an iterative solver.

    Args:
        func: a function to optimize
        opt: an iterative solver
        has_aux: option propagated to jax.grad
        params_predicate: predicate function to partition parameters in trainable parameter
            and non-trainable parameters.
        grads_fill_nan_inf: if True, fill nan and infinite value of the gradients with zeros.
        name: name of the module.

    Returns:
        results: return of func
        trainable_params: updated trainable params
    """

    def __init__(
        self,
        func,
        opt,
        has_aux=False,
        params_predicate: Optional[
            Callable[[str, str, jnp.ndarray], bool]
        ] = pass_all_predicate,
        grads_fill_nan_inf=False,
        name=None,
    ):
        self.func = func
        self.opt = opt
        self.has_aux = has_aux
        self.grads_fill_nan_inf = grads_fill_nan_inf
        self.params_predicate = params_predicate
        super().__init__(name=name)

    def __call__(self, *args, **kwargs):
        trainable_params = hk.get_state(
            "trainable_params",
            [],
            init=lambda *_: get_init_params(
                self.func, params_predicate=self.params_predicate, *args, **kwargs
            ),
        )
        func = UpdateParams(self.func, params_predicate=self.params_predicate)
        results, grads = jax.value_and_grad(func, has_aux=self.has_aux)(
            trainable_params, *args, **kwargs
        )
        if self.grads_fill_nan_inf:
            grads = FillNanInf()(grads)

        # trainable_params = jax.tree_util.tree_map(self.opt, trainable_params, grads)
        trainable_params = self.opt(trainable_params, grads)

        hk.set_state("trainable_params", trainable_params)
        return results, trainable_params
