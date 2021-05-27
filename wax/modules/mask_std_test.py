import haiku as hk
import jax
import jax.numpy as jnp

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


def test_mask_std_with_mask():

    rng = hk.PRNGSequence(42)
    x = jax.random.normal(next(rng), (10,))
    x_std_ref = x[1:].std()

    fun = hk.transform(lambda mask, x: MaskStd()(mask, x))

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
