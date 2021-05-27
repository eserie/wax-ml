import haiku as hk
import jax
import jax.numpy as jnp

from wax.modules.fill_nan_inf import FillNanInf


def test_fill_nan_inf():

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(100, 5), key=next(seq), dtype=jnp.float64)
    mask_nan = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask_nan, x, jnp.nan)
    del mask_nan

    mask_inf = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask_inf, x, jnp.inf)
    del mask_inf

    mask_ninf = jax.random.choice(next(seq), 2, shape=x.shape)
    x = jnp.where(mask_ninf, x, -jnp.inf)
    del mask_ninf

    fun = hk.transform(lambda x: FillNanInf()(x))

    params = fun.init(next(seq), x)
    x_fill = fun.apply(params, next(seq), x)
    assert jnp.allclose(x_fill, jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
