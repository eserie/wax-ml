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

from wax.modules.mask_normalize import MaskNormalize


def normalize(x):
    return x / x.std()


def test_mask_normalize_no_mask():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))

    fun = hk.transform(lambda mask, x: MaskNormalize()(mask, x))
    mask = jnp.full(x.shape, True)

    params = fun.init(next(rng), mask, x)
    x_normalized = fun.apply(params, next(rng), mask, x)
    x_normalized_ref = normalize(x)

    assert jnp.allclose(x_normalized, x_normalized_ref)


def test_mask_normalize_with_mask():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))
    fun = hk.transform(lambda mask, x: MaskNormalize()(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[:1], False)

    params = fun.init(next(rng), mask, x)
    x_normalized = fun.apply(params, next(rng), mask, x)
    x_normalized_ref = normalize(x[1:])
    assert jnp.allclose(x_normalized[1:], x_normalized_ref)


def test_mask_normalize_with_mask_shape_2():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    fun = hk.transform(lambda mask, x: MaskNormalize()(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[0, 2], False)

    params = fun.init(next(rng), mask, x)
    x_normalized = fun.apply(params, next(rng), mask, x)

    # x_normalized_ref = jax.ops.index_update(x, ~mask, 0).normalize()
    x_normalized_ref = x / x[mask].std()
    assert jnp.allclose(x_normalized, x_normalized_ref)


def test_mask_normalize_axis_1():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    fun = hk.transform(lambda mask, x: MaskNormalize(axis=0)(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[0, 2], False)

    params = fun.init(next(rng), mask, x)
    x_normalized = fun.apply(params, next(rng), mask, x)

    # the last axis has only one observation
    assert jnp.allclose(x_normalized[:, 2], 0.0)

    for col in range(3):
        x_normalized_col_ref = jnp.nan_to_num(
            normalize(x[:, col][mask[:, col]]), posinf=0.0
        )
        assert jnp.allclose(x_normalized[:, col], x_normalized_col_ref)


##############################
# test normalize
##############################
