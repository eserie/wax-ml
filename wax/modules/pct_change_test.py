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
import jax
import pytest
from jax import numpy as jnp

from wax.compile import jit_init_apply
from wax.modules.pct_change import PctChange


@pytest.mark.parametrize("use_jit", [False, True])
def test_pct_change(use_jit):
    seq = hk.PRNGSequence(42)
    x = jax.random.normal(next(seq), (2, 3))

    @hk.transform_with_state
    def pct_change(x):
        return PctChange(1)(x)

    if use_jit:
        pct_change = jit_init_apply(pct_change)
    params, state = pct_change.init(next(seq), x)
    output, state = pct_change.apply(params, state, next(seq), x)
    assert len(output) == 2
    assert jnp.isnan(output).all()

    x1 = x
    x = jax.random.normal(next(seq), (2, 3))
    output, state = pct_change.apply(params, state, next(seq), x)
    assert len(output) == 2
    assert (output == (x / x1 - 1.0)).all()
