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
from wax.unroll import unroll


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_init_and_first_step_cov_float64(dtype):
    if dtype == "float64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)
    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(3,), key=next(seq), dtype=jnp.float64)
    y = jax.random.normal(shape=(3,), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x, y):
        return EWMCov(com=10, adjust=True)(x, y)

    params, state = model.init(next(seq), x, y)
    cov, state = model.apply(params, state, next(seq), x, y)
    assert cov.dtype == jnp.dtype(dtype)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_vs_sklearn(assume_centered):
    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x

    com = 1.0e6

    @jit_init_apply
    @hk.transform_with_state
    def model(x, y):
        return EWMCov(com=com, adjust=True, assume_centered=assume_centered)(x, y)

    cov = unroll(model)(x, y)
    cov_ref = EmpiricalCovariance(assume_centered=assume_centered).fit(x).covariance_

    assert jnp.allclose(cov[-1], cov_ref, atol=1.0e-6)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_vs_sklearn_adjust(assume_centered):
    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x

    com = 1.0e6

    @jit_init_apply
    @hk.transform_with_state
    def model(x, y):
        return EWMCov(com=com, adjust=True, assume_centered=assume_centered)(x, y)

    cov = unroll(model)(x, y)
    cov_ref = EmpiricalCovariance(assume_centered=assume_centered).fit(x).covariance_

    assert jnp.allclose(cov[-1], cov_ref, atol=1.0e-6)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_vs_pandas_adjust_finite(assume_centered):
    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x

    com = 1.0e6

    @jit_init_apply
    @hk.transform_with_state
    def model(x, y):
        return EWMCov(com=com, adjust="linear", assume_centered=assume_centered)(x, y)

    cov = unroll(model)(x, y)
    cov_ref = EmpiricalCovariance(assume_centered=assume_centered).fit(x).covariance_

    assert jnp.allclose(cov[-1], cov_ref, atol=1.0e-6)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_run_cov_with_legacy_api(assume_centered):
    config.update("jax_enable_x64", True)

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)
    y = x

    com = 10

    @jit_init_apply
    @hk.transform_with_state
    def model(x, y):
        return EWMCov(com=com, adjust="linear", assume_centered=assume_centered)(x, y)

    cov = unroll(model)(x, y)

    @jit_init_apply
    @hk.transform_with_state
    def model(x_y):
        return EWMCov(com=com, adjust="linear", assume_centered=assume_centered)(x_y)

    cov_ref = unroll(model)(x_y=(x, y))

    assert jnp.allclose(cov, cov_ref, atol=1.0e-6)
