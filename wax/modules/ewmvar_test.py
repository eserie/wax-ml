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
from wax.modules.ewma import EWMA
from wax.modules.ewmvar import EWMVar
from wax.unroll import dynamic_unroll


# Another implementation for checking
class EWMVar_v2(hk.Module):
    """Exponentially weighted variance.
    To calculate the variance we use the fact that Var(X) = Mean(x^2) - Mean(x)^2 and internally
    we use the exponentially weighted mean of x/x^2 to calculate this.

    Arguments:
        alpha : The closer `alpha` is to 1 the more the statistic will adapt to recent values.

    Attributes:
        variance : The running exponentially weighted variance.

    References
    ----------
    [^1]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf) # noqa
    """

    def __init__(self, alpha=0.5, adjust=True, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.adjust = adjust

    def __call__(self, x):
        mean = EWMA(self.alpha, self.adjust, initial_value=jnp.nan, name="mean")(x)
        mean_square = EWMA(
            self.alpha, self.adjust, initial_value=jnp.nan, name="mean_square"
        )(x * x)
        var = mean_square - mean ** 2
        var = jnp.where(var < 0, 0.0, var)
        return var


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_init_and_first_step_var_float64(dtype):
    from jax.config import config

    if dtype == "float64":
        config.update("jax_enable_x64", True)
    else:
        config.update("jax_enable_x64", False)
    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(3,), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMVar(0.1, adjust=True)(x)

    params, state = model.init(next(seq), x)
    var, state = model.apply(params, state, next(seq), x)
    assert var.dtype == jnp.dtype(dtype)


def test_run_var_vs_pandas_not_adjust():
    from jax.config import config

    config.update("jax_enable_x64", True)
    import pandas as pd

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMVar(0.1, adjust=False)(x)

    var, state = dynamic_unroll(model, None, None, next(seq), False, x)
    var = pd.DataFrame(var)

    @jit_init_apply
    @hk.transform_with_state
    def model2(x):
        return EWMVar_v2(0.1, adjust=False)(x)

    var2, state2 = dynamic_unroll(model2, None, None, next(seq), False, x)
    var2 = pd.DataFrame(var2)
    assert jnp.allclose(var, var2)

    pandas_var = pd.DataFrame(x).ewm(alpha=0.1, adjust=False).var()
    assert not jnp.allclose(var, pandas_var.values)


def test_run_var_vs_pandas_adjust():
    from jax.config import config

    config.update("jax_enable_x64", True)
    import pandas as pd

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMVar(0.1, adjust=True)(x)

    var, state = dynamic_unroll(model, None, None, next(seq), False, x)
    var = pd.DataFrame(var)

    @jit_init_apply
    @hk.transform_with_state
    def model2(x):
        return EWMVar_v2(0.1, adjust=True)(x)

    var2, state2 = dynamic_unroll(model2, None, None, next(seq), False, x)
    var2 = pd.DataFrame(var2)
    assert jnp.allclose(var, var2)

    # pandas does something else
    pandas_var = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).var()
    assert not jnp.allclose(var, pandas_var.values)


def test_run_var_vs_pandas_adjust_finite():
    from jax.config import config

    config.update("jax_enable_x64", True)
    import pandas as pd

    seq = hk.PRNGSequence(42)
    x = jax.random.normal(shape=(10, 3), key=next(seq), dtype=jnp.float64)

    @jit_init_apply
    @hk.transform_with_state
    def model(x):
        return EWMVar(0.1, adjust="linear")(x)

    var, state = dynamic_unroll(model, None, None, next(seq), False, x)
    var = pd.DataFrame(var)

    @jit_init_apply
    @hk.transform_with_state
    def model2(x):
        return EWMVar_v2(0.1, adjust=True)(x)

    var2, state2 = dynamic_unroll(model2, None, None, next(seq), False, x)
    var2 = pd.DataFrame(var2)
    assert not jnp.allclose(var, var2)
    # TODO: understand why the two implementations are
    #  not agree for "linear" adjustement scheme.

    pandas_var_adjust = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).var()
    pandas_var_not_adjust = pd.DataFrame(x).ewm(alpha=0.1, adjust=True).var()
    assert not jnp.allclose(var, pandas_var_adjust.values)
    assert not jnp.allclose(var, pandas_var_not_adjust.values)
    corr = jnp.corrcoef(
        var.fillna(0).values.flatten(), pandas_var_adjust.fillna(0).values.flatten()
    )[0, 1]
    assert 0.08 < 1 - corr < 0.1
