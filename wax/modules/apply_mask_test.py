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

from wax.modules.apply_mask import ApplyMask


def test_apply_mask_axis_0():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    mask = jnp.full(x.shape[0], True)
    mask = jax.ops.index_update(mask, 0, False)

    # check that an error is raised if axis is not specified
    fun = hk.transform(lambda mask, x: ApplyMask()(mask, x))
    with pytest.raises(ValueError):
        fun.init(next(rng), mask, x)

    # check that specifying the wrong axis raises
    fun = hk.transform(lambda mask, x: ApplyMask(axis=1)(mask, x))
    with pytest.raises(ValueError):
        fun.init(next(rng), mask, x)

    # now specify axis
    fun = hk.transform(lambda mask, x: ApplyMask(axis=0)(mask, x))
    params = fun.init(next(rng), mask, x)
    x_mask = fun.apply(params, next(rng), mask, x)

    assert jnp.allclose(x_mask[0, :], 0.0)


def test_apply_mask_axis_1():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3))

    mask = jnp.full(x.shape[1], True)
    mask = jax.ops.index_update(mask, 0, False)

    # check that an error is raised if axis is not specified
    fun = hk.transform(lambda mask, x: ApplyMask()(mask, x))
    with pytest.raises(ValueError):
        fun.init(next(rng), mask, x)

    # check that specifying the wrong axis raises
    fun = hk.transform(lambda mask, x: ApplyMask(axis=0)(mask, x))
    with pytest.raises(ValueError):
        fun.init(next(rng), mask, x)

    # now specify axis
    fun = hk.transform(lambda mask, x: ApplyMask(axis=1)(mask, x))
    params = fun.init(next(rng), mask, x)
    x_mask = fun.apply(params, next(rng), mask, x)

    assert jnp.allclose(x_mask[:, 0], 0.0)


def test_apply_mask_axis_2_raises():
    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (2, 3, 1))

    mask = jnp.full(x.shape[1], True)
    mask = jax.ops.index_update(mask, 0, False)

    # check that an error is raised if axis is not specified
    fun = hk.transform(lambda mask, x: ApplyMask(axis=2)(mask, x))
    with pytest.raises(ValueError) as err:
        fun.init(next(rng), mask, x)
    assert (
        str(err.value)
        == "ApplyMask is not implemented for axis different from None, 0, 1."
    )
