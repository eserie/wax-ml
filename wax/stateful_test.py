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
from wax.stateful import unroll_lift_with_state, vmap_lift_with_state
import haiku  as hk
import jax
import jax.numpy as jnp


class MyModule(hk.Module):
    def __init__(self, steps, name=None):
        super().__init__(name=name)
        self.steps = steps

    def __call__(self, x):
        assert x.ndim == 0
        steps = hk.get_parameter("steps", (1,), init=lambda *_: self.steps*jnp.ones(1))
        state = hk.get_state("state", (1,), init=lambda *_: jnp.zeros(1) + x)
        state = state + steps
        hk.set_state("state", state)
        return state + x


def test_vmpa_lift_state():

    x = jnp.arange(3).astype(jnp.float32)

    def run_vmap():
        def outer_fun(x):
            def fun(x):
                return MyModule(steps=2)(x)
            return vmap_lift_with_state(fun)(x)

        init, apply = hk.transform_with_state(outer_fun)
        params, state = init(jax.random.PRNGKey(0), x)
        out, state = apply(params, state, None, x)
        out, state = apply(params, state, None, x)
        return out

    def run_static():
        steps = 2
        state = jnp.zeros_like(x) + x

        # apply 1
        state = state + steps
        out = state + x
        # apply 2
        state = state + steps
        out = state + x
        return out.reshape(-1, 1)
    assert jnp.allclose(run_vmap(), run_static())


def test_unroll_lift_state():
    x = jnp.zeros(3).astype(jnp.float32)
    def run_unroll():

        def outer_fun(x):
            def fun(x):
                return MyModule(steps=1)(x)
            return unroll_lift_with_state(fun)(x)

        init, apply = hk.transform_with_state(outer_fun)
        params, state = init(jax.random.PRNGKey(0), x)
        out, state = apply(params, state, None, x)
        out, state = apply(params, state, None, x)
        return out

    def run_static():
        steps = 1
        state = jnp.zeros(1) + x[0]
        out = []
        for i in range(len(x)):
            state = state + steps
            out.append(state + x[i])
        out = []
        for i in range(len(x)):
            state = state + steps
            out.append(state + x[i])
        out = jnp.stack(out)
        return out
    assert jnp.allclose(run_unroll(), run_static())
