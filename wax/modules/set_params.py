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
import haiku as hk


class SetParams(hk.Module):
    def __init__(self, fun, name=None):
        super().__init__(name=name)
        self.fun = (
            fun
            if isinstance(fun, hk.TransformedWithState)
            else hk.transform_with_state(fun)
        )

    def __call__(self, params, *args, **kwargs):
        rng = hk.next_rng_key()

        def init_state(*_):
            _, state = self.fun.init(rng, *args, **kwargs)
            return state

        state = hk.get_state("state", [], init=init_state)
        res, state = self.fun.apply(params, state, rng, *args, **kwargs)
        hk.set_state("state", state)

        return res
