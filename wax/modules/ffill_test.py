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
import pytest
from jax import numpy as jnp

from wax.modules import Ffill
from wax.unroll import unroll


@pytest.mark.parametrize("use_jit", [False, True])
def test_ffill(use_jit):
    xs = jnp.array([90, 91, jnp.nan, 85])
    res = unroll(lambda x: Ffill()(x))(xs)
    assert jnp.allclose(res, jnp.array([90, 91, 91, 85], dtype=jnp.float32))
