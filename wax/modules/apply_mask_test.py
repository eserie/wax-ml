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
