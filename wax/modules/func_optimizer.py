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
import haiku as hk
import jax

from wax.modules import FillNanInf
from wax.modules.transform_params import TransformParams, get_init_params


class FuncOptimizer(hk.Module):
    def __init__(self, func, opt, has_aux=False, grads_fill_nan_inf=False, name=None):
        self.func = func
        self.opt = opt
        self.has_aux = has_aux
        self.grads_fill_nan_inf = grads_fill_nan_inf
        super().__init__(name=name)

    def __call__(self, *args, **kwargs):
        params = hk.get_state(
            "params", [], init=lambda *_: get_init_params(self.func, *args, **kwargs)
        )
        func_params = TransformParams(self.func)
        l, grads = jax.value_and_grad(func_params, has_aux=self.has_aux)(
            params, *args, **kwargs
        )
        if self.grads_fill_nan_inf:
            grads = FillNanInf()(grads)

        params = jax.tree_multimap(self.opt, params, grads)
        hk.set_state("params", params)
        return l, params
