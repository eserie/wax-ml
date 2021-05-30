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

from wax.modules.mask_mean import MaskMean


def test_mask_mean_no_mask():

    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))
    x_mean_ref = x.mean()

    fun = hk.transform(lambda mask, x: MaskMean()(mask, x))
    mask = jnp.full(x.shape, True)

    params = fun.init(next(rng), mask, x)
    x_mean = fun.apply(params, next(rng), mask, x)
    assert jnp.allclose(x_mean, x_mean_ref)


def test_mask_mean_with_mask():

    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))
    x_mean_ref = x[1:].mean()

    fun = hk.transform(lambda mask, x: MaskMean()(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[:1], False)

    params = fun.init(next(rng), mask, x)
    x_mean = fun.apply(params, next(rng), mask, x)
    assert jnp.allclose(x_mean, x_mean_ref)


def test_mask_mean_with_mask_shape_2():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    fun = hk.transform(lambda mask, x: MaskMean()(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[0, 2], False)

    params = fun.init(next(rng), mask, x)
    x_mean = fun.apply(params, next(rng), mask, x)

    # mean_ref = jax.ops.index_update(x, ~mask, 0).mean()
    x_mean_ref = x[mask].mean()
    assert jnp.allclose(x_mean, x_mean_ref)


def test_mask_mean_axis_1():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    fun = hk.transform(lambda mask, x: MaskMean(axis=0)(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[0, 2], False)

    params = fun.init(next(rng), mask, x)
    x_mean = fun.apply(params, next(rng), mask, x)

    # the last axis has only one observation
    assert jnp.allclose(x_mean[2], x[1, 2])

    for col in range(3):
        x_mean_col_ref = x[:, col][mask[:, col]].mean()
        assert jnp.allclose(x_mean[col], x_mean_col_ref)
