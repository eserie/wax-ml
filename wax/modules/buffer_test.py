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

from wax.compile import jit_init_apply
from wax.modules.buffer import Buffer


@pytest.mark.parametrize("use_jit", [False, True])
def test_buffer_dict(use_jit):
    seq = hk.PRNGSequence(42)
    x = {
        "x": jax.random.normal(next(seq), (2, 3)),
        "y": jax.random.normal(next(seq), (3, 4)),
    }

    @hk.transform_with_state
    def buffer(x):
        return Buffer(2, return_state=False, name="buf")(x)

    if use_jit:
        buffer = jit_init_apply(buffer)

    params, state = buffer.init(next(seq), x)
    assert jnp.isnan(state["buf"]["buffer_state"].buffer["x"]).all()
    assert jnp.isnan(state["buf"]["buffer_state"].buffer["y"]).all()
    assert state["buf"]["buffer_state"].len_buffer == 0
    assert state["buf"]["buffer_state"].i_start == 2

    output, state = buffer.apply(params, state, next(seq), x)


@pytest.mark.parametrize("use_jit", [False, True])
def test_buffer_dict_fillna_struct(use_jit):
    seq = hk.PRNGSequence(42)
    x = {
        "x": jax.random.normal(next(seq), (2, 3)),
        "y": jax.random.normal(next(seq), (3, 4)),
    }

    fill_value = {"x": -1, "y": 999}

    @hk.transform_with_state
    def buffer(x):
        return Buffer(2, return_state=False, name="buf", fill_value=fill_value)(x)

    if use_jit:
        buffer = jit_init_apply(buffer)

    params, state = buffer.init(next(seq), x)
    assert (state["buf"]["buffer_state"].buffer["x"] == -1).all()
    assert (state["buf"]["buffer_state"].buffer["y"] == 999).all()
    assert state["buf"]["buffer_state"].len_buffer == 0
    assert state["buf"]["buffer_state"].i_start == 2

    output, state = buffer.apply(params, state, next(seq), x)


@pytest.mark.parametrize("use_jit", [False, True])
def test_buffer(use_jit):
    seq = hk.PRNGSequence(42)
    x = jax.random.normal(next(seq), (2, 3))

    @hk.transform_with_state
    def buffer(x):
        return Buffer(2, return_state=False, name="buf")(x)

    if use_jit:
        buffer = jit_init_apply(buffer)

    params, state = buffer.init(next(seq), x)
    assert jnp.isnan(state["buf"]["buffer_state"].buffer).all()
    assert state["buf"]["buffer_state"].len_buffer == 0
    assert state["buf"]["buffer_state"].i_start == 2

    output, state = buffer.apply(params, state, next(seq), x)
    assert len(output) == 2
    if use_jit:
        assert state["buf"]["buffer_state"].buffer is not output
        assert jnp.allclose(state["buf"]["buffer_state"].buffer, output, equal_nan=True)
    else:
        assert state["buf"]["buffer_state"].buffer is output
    assert state["buf"]["buffer_state"].len_buffer == 1
    assert state["buf"]["buffer_state"].i_start == 1
    assert jnp.allclose(output[-1], x)
    assert jnp.isnan(output[-2]).all()
    x1 = x

    x = jax.random.normal(next(seq), (2, 3))
    output, state = buffer.apply(params, state, next(seq), x)
    assert len(output) == 2
    if use_jit:
        assert state["buf"]["buffer_state"].buffer is not output
        assert jnp.allclose(state["buf"]["buffer_state"].buffer, output, equal_nan=True)
    else:
        assert state["buf"]["buffer_state"].buffer is output
    assert state["buf"]["buffer_state"].len_buffer == 2
    assert state["buf"]["buffer_state"].i_start == 0
    assert jnp.allclose(output[-1], x)
    assert jnp.allclose(output[-2], x1)
    x2 = x

    x = jax.random.normal(next(seq), (2, 3))
    output, state = buffer.apply(params, state, next(seq), x)
    assert len(output) == 2
    if use_jit:
        assert state["buf"]["buffer_state"].buffer is not output
        assert jnp.allclose(state["buf"]["buffer_state"].buffer, output, equal_nan=True)
    else:
        assert state["buf"]["buffer_state"].buffer is output
    assert state["buf"]["buffer_state"].len_buffer == 2
    assert state["buf"]["buffer_state"].i_start == 0
    assert jnp.allclose(output[-1], x)
    assert jnp.allclose(output[-2], x2)
