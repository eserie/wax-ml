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
import numpy as onp
import pandas as pd
import pytest

from wax.external.eagerpy import convert_to_tensors
from wax.universal.eager_ewma import (
    EagerEWMA,
    dynamic_unroll_fori_loop_universal,
    dynamic_unroll_tf,
    dynamic_unroll_universal,
    static_unroll_universal,
)


def test_init_and_first_step_ema_float64():
    from jax.config import config

    config.update("jax_enable_x64", True)
    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(3,), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust=True)(x)

    params, state = model.init(key, x)
    ema, state = model.apply(params, state, key, x)
    assert ema.dtype == jnp.dtype("float64")


def test_init_and_first_step_ema():
    from jax.config import config

    config.update("jax_enable_x64", False)

    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(3,), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust=True)(x)

    params, state = model.init(key, x)
    ema, state1 = model.apply(params, state, key, x)
    assert ema.dtype == jnp.dtype("float32")


def test_run_ema_vs_pandas_not_adjust():
    from jax.config import config

    config.update("jax_enable_x64", True)
    import pandas as pd

    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(100, 3), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust=False)(x)

    ema, state = static_unroll_universal(model, None, None, key, False, x)

    pandas_ema = pd.DataFrame(x).ewm(alpha=jnp.array(0.1), adjust=False).mean()

    assert jnp.allclose(ema, pandas_ema.values)


def test_dynamic_unroll_fori_loop():
    from jax.config import config

    config.update("jax_enable_x64", True)

    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(100, 3), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust=True)(x)

    ema, state = static_unroll_universal(model, None, None, key, False, x)

    ema2, state2 = dynamic_unroll_fori_loop_universal(model, None, None, key, False, x)

    assert jnp.allclose(ema, ema2)


def test_dynamic_unroll():
    from jax.config import config

    config.update("jax_enable_x64", True)

    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(100, 3), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust=True)(x)

    ema, state = static_unroll_universal(model, None, None, key, False, x)

    ema2, state2 = dynamic_unroll_universal(model, None, None, key, False, x)

    assert jnp.allclose(ema, ema2)


def test_run_ema_vs_pandas_adjust():
    from jax.config import config

    config.update("jax_enable_x64", True)

    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(100, 3), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust=True)(x)

    ema, state = dynamic_unroll_universal(model, None, None, key, False, x)

    pandas_ema = pd.DataFrame(x).ewm(alpha=jnp.array(0.1), adjust=True).mean()
    assert jnp.allclose(ema, pandas_ema.values)


def test_run_ema_vs_pandas_adjust_finite():
    from jax.config import config

    config.update("jax_enable_x64", True)

    seed = 1701
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(shape=(100, 3), key=key, dtype=jnp.float64)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(0.1), adjust="linear")(x)

    ema, state = dynamic_unroll_universal(model, None, None, key, False, x)

    pandas_ema_adjust = pd.DataFrame(x).ewm(alpha=jnp.array(0.1), adjust=True).mean()
    pandas_ema_not_adjust = (
        pd.DataFrame(x).ewm(alpha=jnp.array(0.1), adjust=True).mean()
    )
    assert not jnp.allclose(ema, pandas_ema_adjust.values)
    assert not jnp.allclose(ema, pandas_ema_not_adjust.values)
    corr = jnp.corrcoef(ema.flatten(), pandas_ema_adjust.values.flatten())[0, 1]
    assert 1.0e-3 < 1 - corr < 1.0e-2


@pytest.mark.parametrize("tensor_type", ["numpy", "jax", "tensorflow", "torch"])
def test_backends(tensor_type):
    from jax.config import config

    config.update("jax_enable_x64", True)

    key = jax.random.PRNGKey(403)
    ox = onp.random.normal(size=(100, 3))
    oalpha = onp.array(0.1)
    x, alpha = convert_to_tensors((ox, oalpha), tensor_type)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(alpha, adjust=False)(x)

    ema, state = static_unroll_universal(model, None, None, key, False, x)

    pandas_ema = pd.DataFrame(ox).ewm(oalpha, adjust=False).mean()
    df = pd.concat(
        [pd.DataFrame(ox), pd.DataFrame(ema), pandas_ema],
        keys=["input", tensor_type, "pandas"],
        axis=1,
    ).stack()
    df = df.xs(0, level=1)
    assert len(df)
    assert not jnp.allclose(ema, pandas_ema.values)


def test_tf_optimized():
    tensor_type = "tensorflow"

    key = None  # jax.random.PRNGKey(403)
    ox = onp.random.normal(size=(100, 3))
    oalpha = 0.1
    x, alpha = convert_to_tensors((ox, oalpha), tensor_type)

    @hk.transform_with_state
    def model(x):
        return EagerEWMA(jnp.array(alpha), adjust=False)(x)

    ema, state = dynamic_unroll_tf(model, None, None, key, False, x)

    pandas_ema = pd.DataFrame(ox).ewm(oalpha, adjust=False).mean()
    df = pd.concat(
        [pd.DataFrame(ox), pd.DataFrame(ema), pandas_ema],
        keys=["input", tensor_type, "pandas"],
        axis=1,
    ).stack()
    df = df.xs(0, level=1)
    assert len(df)
    assert not jnp.allclose(ema, pandas_ema.values)
