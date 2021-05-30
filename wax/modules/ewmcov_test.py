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
import jax.numpy as jnp
import pytest
from jax.config import config
from sklearn.covariance import EmpiricalCovariance

from wax.compile import jit_init_apply
from wax.modules.ewmcov import EWMCov
from wax.unroll import dynamic_unroll

# Another implementation for checking


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_init_and_first_step_cov_float64(dtype):

    if dtype == "float64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)
    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(3,), key=next(seq), dtype=jnp.float64)
    y = jax.random.normal(shape=(3,), key=next(seq), dtype=jnp.float64)
    data = (x, y)

    @jit_init_apply
    @hk.transform_with_state
    def model(data):

        return EWMCov(jnp.array(0.1), adjust=True)(data)

    params, state = model.init(next(seq), data)
    cov, state = model.apply(params, state, next(seq), data)
    assert cov.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_vs_sklearn(assume_centered):

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x
    data = (x, y)

    alpha = 1.0e-6

    @jit_init_apply
    @hk.transform_with_state
    def model(data):
        return EWMCov(alpha, adjust=True, assume_centered=assume_centered)(data)

    cov, state = dynamic_unroll(model, None, None, next(seq), False, data)

    cov_ref = EmpiricalCovariance(assume_centered=assume_centered).fit(x).covariance_

    assert jnp.allclose(cov[-1], cov_ref, atol=1.0e-6)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_vs_sklearn_adjust(assume_centered):

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x
    data = (x, y)
    alpha = 1.0e-6

    @jit_init_apply
    @hk.transform_with_state
    def model(data):

        return EWMCov(alpha, adjust=True, assume_centered=assume_centered)(data)

    cov, state = dynamic_unroll(model, None, None, next(seq), False, data)
    cov_ref = EmpiricalCovariance(assume_centered=assume_centered).fit(x).covariance_

    assert jnp.allclose(cov[-1], cov_ref, atol=1.0e-6)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_vs_pandas_adjust_finite(assume_centered):

    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x
    data = (x, y)

    alpha = 1.0e-6

    @jit_init_apply
    @hk.transform_with_state
    def model(data):
        return EWMCov(alpha, adjust="linear", assume_centered=assume_centered)(data)

    cov, state = dynamic_unroll(model, None, None, next(seq), False, data)
    cov_ref = EmpiricalCovariance(assume_centered=assume_centered).fit(x).covariance_

    assert jnp.allclose(cov[-1], cov_ref, atol=1.0e-6)
