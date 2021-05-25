# Copyright 2021 The Wax Authors
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

from wax.modules.rolling_mean import RollingMean


def test_rolling_mean():
    @hk.transform_with_state
    def mean(x):
        return RollingMean(2)(x)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(next(seq), (2, 3))
    params, state = mean.init(next(seq), x)
    output, state = mean.apply(params, state, next(seq), x)
    assert output is not x
    assert (output == x).all()
    x1 = x

    x = jax.random.normal(next(seq), (2, 3))
    output, state = mean.apply(params, state, next(seq), x)
    assert ((x1 + x) / 2 == output).all()
    x2 = x

    x = jax.random.normal(next(seq), (2, 3))
    output, state = mean.apply(params, state, next(seq), x)
    assert ((x2 + x) / 2 == output).all()
