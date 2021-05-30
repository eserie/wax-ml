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

from wax.modules.mask_std import MaskStd


def test_mask_std_no_mask():

    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))
    std_ref = x.std()

    fun = hk.transform(lambda mask, x: MaskStd()(mask, x))
    mask = jnp.full(x.shape, True)

    params = fun.init(next(rng), mask, x)
    x_std = fun.apply(params, next(rng), mask, x)
    assert jnp.allclose(x_std, std_ref)


@pytest.mark.parametrize("assume_centered", [False, True])
def test_mask_std_with_mask(assume_centered):

    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))
    if assume_centered:
        x_std_ref = jnp.sqrt((x[1:] ** 2).mean())
    else:
        x_std_ref = x[1:].std()

    fun = hk.transform(
        lambda mask, x: MaskStd(assume_centered=assume_centered)(mask, x)
    )

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[:1], False)

    params = fun.init(next(rng), mask, x)
    x_std = fun.apply(params, next(rng), mask, x)
    assert jnp.allclose(x_std, x_std_ref)


def test_mask_std_with_mask_shape_2():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    fun = hk.transform(lambda mask, x: MaskStd()(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[0, 2], False)

    params = fun.init(next(rng), mask, x)
    x_std = fun.apply(params, next(rng), mask, x)

    # std_ref = jax.ops.index_update(x, ~mask, 0).std()
    x_std_ref = x[mask].std()
    assert jnp.allclose(x_std, x_std_ref)


def test_mask_std_axis_1():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    fun = hk.transform(lambda mask, x: MaskStd(axis=0)(mask, x))

    mask = jnp.full(x.shape, True)
    mask = jax.ops.index_update(mask, jax.ops.index[0, 2], False)

    params = fun.init(next(rng), mask, x)
    x_std = fun.apply(params, next(rng), mask, x)

    # the last axis has only one observation
    assert jnp.allclose(x_std[2], 0.0)

    for col in range(3):
        x_std_col_ref = x[:, col][mask[:, col]].std()
        assert jnp.allclose(x_std[col], x_std_col_ref)
