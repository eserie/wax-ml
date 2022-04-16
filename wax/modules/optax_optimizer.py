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
import optax


class OptaxOptimizer(hk.Module):
    """A module which wraps an optax GrandientTransformation.

    Args:
        opt: gradient transformation
        name: name of the module
    """

    def __init__(self, opt: optax.GradientTransformation, name=None):
        super().__init__(name=name)
        self.opt = opt

    def __call__(self, params, grads):

        # force conversion to immutable dict
        # In jupyter notebooks the following exception is raised:
        # ValueError: Custom node type mismatch: expected type
        # when used with optax.adam optimizer, the opt.init(params) appears
        # to return a dict instead of a FlatMap.
        # Strangely, this error does not appear in tests.
        params = hk.data_structures.to_immutable_dict(params)
        grads = hk.data_structures.to_immutable_dict(grads)

        opt_state = hk.get_state("opt_state", [], init=lambda *_: self.opt.init(params))
        updates, opt_state = self.opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        hk.set_state("opt_state", opt_state)
        return params
