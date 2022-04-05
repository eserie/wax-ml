# Copyright 2022 The WAX-ML Authors
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
    (l_, yp), w_history = res

    # check that averaged loss is less than initial loss.
    assert l_.mean() < l_[0] / 37
    # pd.DataFrame(w_history["linear"]["w"].squeeze()).plot()
    # plt.show();
    # pd.Series(l).expanding().mean().plot(title="loss")
