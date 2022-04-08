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
from typing import Callable, Optional

import haiku as hk


class UpdateParams(hk.Module):
    """Transform a callable as a function of its trainable parameters.

    The non trainable parameters and state are captured and managed as state
    of the module.
    """

    def __init__(
        self, func: Callable, split_params: Optional[Callable] = None, name=None
    ):
        super().__init__(name=name)
        self.func = func
        self.split_params = (
            split_params
            if split_params is not None
            else lambda params: (params, type(params)())
        )

    def __call__(self, trainable_params, *args, **kwargs):
        init, apply = hk.transform_with_state(self.func)
        rng = hk.next_rng_key() if hk.running_init() else None

        def init_model_non_trainable_params_and_state(shape, dtype):
            """Set state from trainable params and state of the model."""
            params, state = init(rng, *args, **kwargs)
            trainable_params, non_trainable_params = self.split_params(params)
            trainable_params = hk.data_structures.to_mutable_dict(trainable_params)
            return (non_trainable_params, state)

        non_trainable_params, state = hk.get_state(
            "non_trainable_params_and_state",
            [],
            init=init_model_non_trainable_params_and_state,
        )

        params = hk.data_structures.merge(trainable_params, non_trainable_params)

        res, state = apply(params, state, rng, *args, **kwargs)

        hk.set_state("non_trainable_params_and_state", (non_trainable_params, state))
        return res


def get_init_params(func, *args, split_params: Optional[Callable] = None, **kwargs):
    init_rng = hk.next_rng_key() if hk.running_init() else None
    init, _ = hk.transform(func)
    params = init(init_rng, *args, **kwargs)

    if split_params:
        trainable_params, non_trainable_params = split_params(params)
        trainable_params = hk.data_structures.to_mutable_dict(trainable_params)
        return trainable_params
    else:
        return params
