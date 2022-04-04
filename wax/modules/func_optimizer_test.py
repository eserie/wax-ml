import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.config import config

from wax.modules.func_optimizer import FuncOptimizer
from wax.modules.optax_optimizer import OptaxOptimizer
from wax.unroll import unroll


def generate_data():
    T = 1000
    rng = jax.random.PRNGKey(42)
    x = jax.random.normal(rng, (T, 3))
    w = jnp.ones((3,)).at[2].set(0)
    y = x @ w + 0.1 * jax.random.normal(rng, (T,))
    return x, y


def test_func_optimizer():
    config.update("jax_enable_x64", False)

    x, y = generate_data()

    def loss(y, yp):
        return jnp.square(y - yp).mean()

    def learn(x, y):
        def cost(x, y):
            yp = hk.Linear(1, with_bias=False)(x)
            return loss(y, yp.reshape(y.shape)), yp

        (l, yp), params = FuncOptimizer(
            cost, OptaxOptimizer(optax.sgd(1.0e-2)), has_aux=True
        )(x, y)

        return (l, yp), params

    rng = jax.random.PRNGKey(42)
    res = unroll(learn, rng=rng)(x, y)
    (loss, yp), w_history = res

    # check that averaged loss is less than initial loss.
    assert loss.mean() < loss[0] / 37
    # pd.DataFrame(w_history["linear"]["w"].squeeze()).plot()
    # plt.show();
    # pd.Series(l).expanding().mean().plot(title="loss")
