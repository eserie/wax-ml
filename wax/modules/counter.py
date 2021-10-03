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
from jax.numpy import array as DeviceArray
from jax.numpy import uint32


class Counter(hk.Module):
    def __call__(self):
        count = hk.get_state(
            "count", [], init=lambda shape, dtype: DeviceArray(0, dtype=uint32)
        )
        count += 1
        hk.set_state("count", count)
        return count
