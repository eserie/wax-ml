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
import jax.numpy as jnp

from wax.compile import jit_init_apply
from wax.modules.update_on_event import UpdateOnEvent

TEST_KEY = jnp.array([255383827, 267815257], dtype=jnp.uint32)


class MyModule(hk.Module):
    def __call__(self, x):
        prev_state = hk.get_state(
            "state",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda shape, dtype: jnp.full(shape, fill_value=0, dtype=dtype),
        )

        state = prev_state
        state = state + x
        output = state * 2

        hk.set_state("state", state)

        return output


def test_no_wrapping():
    @jit_init_apply
    @hk.transform_with_state
    def fun(x):
        return MyModule()(x)

    x = jnp.array(
        [
            -1.6652094,
            0.23443973,
            -0.24498996,
            0.4470721,
            0.25630304,
            0.28259817,
            -0.11017023,
            0.5699526,
            0.33960477,
            -1.7104211,
        ],
        dtype=jnp.float32,
    )
    params, state = fun.init(TEST_KEY, x)

    outputs = []
    for i in range(10):
        output, state = fun.apply(params, state, TEST_KEY, x)
        outputs.append(output)
    outputs = jnp.stack(outputs)
    ref_outputs = jnp.array(
        [
            -3.3304188,
            -6.6608377,
            -9.991257,
            -13.321675,
            -16.652094,
            -19.982513,
            -23.312933,
            -26.643353,
            -29.973772,
            -33.30419,
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(ref_outputs, outputs[:, 0])


def test_deleguate_all_event_true():
    @jit_init_apply
    @hk.transform_with_state
    def fun(event, x):
        return UpdateOnEvent(MyModule())(event, x)

    x = jnp.array(
        [
            -1.6652094,
            0.23443973,
            -0.24498996,
            0.4470721,
            0.25630304,
            0.28259817,
            -0.11017023,
            0.5699526,
            0.33960477,
            -1.7104211,
        ],
        dtype=jnp.float32,
    )

    event = True
    params, state = fun.init(TEST_KEY, event, x)
    outputs = []
    for i in range(10):
        output, state = fun.apply(params, state, TEST_KEY, event, x)
        outputs.append(output)
    outputs = jnp.stack(outputs)
    ref_outputs = jnp.array(
        [
            -3.3304188,
            -6.6608377,
            -9.991257,
            -13.321675,
            -16.652094,
            -19.982513,
            -23.312933,
            -26.643353,
            -29.973772,
            -33.30419,
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(ref_outputs, outputs[:, 0])


def test_deleguate_some_event_true():
    @jit_init_apply
    @hk.transform_with_state
    def fun(event, x):
        return UpdateOnEvent(MyModule(), initial_output_value=-999)(event, x)

    x = jnp.array(
        [
            -1.6652094,
            0.23443973,
            -0.24498996,
            0.4470721,
            0.25630304,
            0.28259817,
            -0.11017023,
            0.5699526,
            0.33960477,
            -1.7104211,
        ],
        dtype=jnp.float32,
    )

    event = False
    params, state = fun.init(TEST_KEY, event, x)

    outputs = []
    for i in range(10):
        if i in [3, 7]:
            event = True
        else:
            event = False
        output, state = fun.apply(params, state, TEST_KEY, event, x)
        outputs.append(output)
    outputs = jnp.stack(outputs)
    ref_outputs = jnp.array(
        [
            -999.0,
            -999.0,
            -999.0,
            -3.3304188,
            -3.3304188,
            -3.3304188,
            -3.3304188,
            -6.6608377,
            -6.6608377,
            -6.6608377,
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(ref_outputs, outputs[:, 0])
    return output
