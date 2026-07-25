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
import jax
import jax.numpy as jnp
import optax

from wax.optim.newton import newton, sherman_morrison


def test_sherman_morrisson():
    x = 1
    A_inv = jnp.eye(3) * 1.0
    u = v = jnp.ones(3)

    ref = jnp.linalg.inv(A_inv + x * jnp.outer(u, v))

    A_inv = sherman_morrison(A_inv, u, v)
    assert jnp.allclose(A_inv, ref)


def loss(w, x):
    return -(w * x).sum() + (w**2).sum()


# from optax import sgd, adagrad


def test_sgd():
    w = jnp.ones(3)
    x = jnp.ones(3)

    l_ = loss(w, x)

    # opt = sgd(1.0e-3)
    # opt = adagrad(1.0e-3)
    opt = newton(1.0e-3)

    l_, grads = jax.value_and_grad(loss)(w, x)

    state = opt.init(w)
    grads, state = opt.update(grads, state)
    w = optax.apply_updates(w, grads)

    assert loss(w, x) < l_
