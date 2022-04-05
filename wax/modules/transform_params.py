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
from typing import Callable

import haiku as hk


class TransformParams(hk.Module):
    """Transform a callable as a function of its parameters.

    The state is captured and managed by the module.
    """

    def __init__(self, func: Callable, name=None):
        super().__init__(name=name)
        self.func = func

    def __call__(self, params, *args, **kwargs):
        init, apply = hk.transform_with_state(self.func)

        rng = hk.next_rng_key() if hk.running_init() else None
        state = hk.get_state(
            "state", [], init=lambda shape, dtype: init(rng, *args, **kwargs)[1]
        )

        res, state = apply(params, state, rng, *args, **kwargs)

        hk.set_state("state", state)
        return res


def get_init_params(func, *args, **kwargs):
    init_rng = hk.next_rng_key() if hk.running_init() else None
    init, _ = hk.transform(func)
    params = init(init_rng, *args, **kwargs)
    return params
