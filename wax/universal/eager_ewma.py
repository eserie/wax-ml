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
"""Universal Exponential moving average module.
"""
from collections import namedtuple
from functools import partial

import eagerpy as ep
import haiku as hk
import jax
import tensorflow as tf
from eagerpy import eager_function
from jax import tree_map
from jax.lax import fori_loop
from tqdm.auto import tqdm


class EagerEWMA(hk.Module):
    """Universal Exponential moving average module."""

    def __init__(self, alpha, adjust=True, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.adjust = adjust

    @eager_function
    def __call__(self, x):

        mean = hk.get_state(
            "mean",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda *_: ep.full_like(x, ep.nan),
        )
        count = hk.get_state(
            "count",
            shape=x.shape,
            dtype=x.dtype,
            init=lambda *_: ep.full_like(x, 0.0),
        )

        alpha = hk.get_parameter(
            "alpha",
            shape=(1,),
            dtype=x.dtype,
            init=lambda *_: type(x)(self.alpha).astype(dtype=x.dtype).reshape(1),
        )

        mean = ep.where(ep.isnan(mean), x, mean)
        count = ep.where(ep.logical_not(ep.isnan(x)), count + 1, count)

        if self.adjust == "linear":
            tscale = 1.0 / alpha
            tscale = ep.where(count < tscale, count, tscale)
            alpha = ep.where(tscale > 0, 1.0 / tscale, ep.nan)

        elif self.adjust:
            # exponential scheme (as in pandas)
            alpha = alpha / (1.0 - (1.0 - alpha) ** (count))

        # update mean  if x is not nan
        mean = ep.where(
            ep.logical_not(ep.isnan(x)),
            alpha * x + (1.0 - alpha) * mean,
            mean,
        )

        hk.set_state("mean", mean)
        hk.set_state("count", count)
        return mean


@eager_function
def static_unroll(fun, x, rng=None, initial_state=None, pbar=True, compile=None):
    rng = rng if rng is not None else next(hk.PRNGSequence(42))

    fun = type(fun)(eager_function(fun.init), eager_function(fun.apply))
    if compile == "jax":
        fun = type(fun)(fun.init, jax.jit(fun.apply))
    if compile == "tensorflow":
        fun = type(fun)(fun.init, tf.function(fun.apply))

    params, fun_init_state = fun.init(rng, x[0])
    initial_state = fun_init_state if initial_state is None else initial_state

    T = len(x)
    state = initial_state
    output_sequence = []

    seq = range(T)
    if pbar:
        seq = tqdm(list(seq))
    for i in seq:
        outputs, state = fun.apply(params, state, rng, x[i])
        output_sequence.append(outputs)
    output_sequence = ep.stack(output_sequence)
    return output_sequence, state


@partial(jax.jit, static_argnums=(0,))
@eager_function
def dynamic_unroll_fori_loop(fun, x, rng=None, initial_state=None):
    LoopState = namedtuple("LoopState", "params, state, rng, x, output_sequence")

    @jax.jit
    def body_fun(i, loop_state):
        params, prev_state, rng, inputs, output_sequence = loop_state
        outputs, next_state = eager_function(fun.apply)(
            params, prev_state, rng, inputs[i]
        )
        output_sequence = ep.index_update(output_sequence, i, outputs)
        return LoopState(
            params,
            next_state,
            rng,
            inputs,
            output_sequence,
        )

    # Initialize loop
    rng = rng if rng is not None else next(hk.PRNGSequence(42))
    params, fun_init_state = eager_function(fun.init)(rng, x[0])
    initial_state = fun_init_state if initial_state is None else initial_state
    output_sequence = ep.full_like(x, ep.nan)
    loop_state = LoopState(params, initial_state, rng, x, output_sequence)

    # run loop
    T = len(x)
    loop_state = fori_loop(0, T, body_fun, loop_state)

    # format results
    final_state = loop_state.state
    output_sequence = loop_state.output_sequence
    return output_sequence, final_state


@partial(jax.jit, static_argnums=(0,))
@eager_function
def dynamic_unroll(fun, x, rng=None, initial_state=None):
    """Implementation with jax.lax.scan.

    We can use scan because the output mean is in the state!

    See : https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    """
    rng = rng if rng is not None else next(hk.PRNGSequence(42))

    def scan_f(prev_state, inputs):
        outputs, next_state = eager_function(fun.apply)(params, prev_state, rng, inputs)
        return next_state, outputs

    params, fun_init_state = eager_function(fun.init)(rng, x[0])
    initial_state = fun_init_state if initial_state is None else initial_state
    final_state, output_sequence = jax.lax.scan(scan_f, init=initial_state, xs=x)
    return output_sequence, final_state


@tf.function
@tf.autograph.experimental.do_not_convert
def dynamic_unroll_tf(fun, x, rng=None, initial_state=None):
    """Implementation with jax.lax.scan.

    We can use scan because the output mean is in the state!

    See : https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    """
    rng = rng if rng is not None else next(hk.PRNGSequence(42))

    def scan_f(prev_state, inputs):
        outputs, next_state = eager_function(fun.apply)(params, prev_state, rng, inputs)
        return next_state

    params, fun_init_state = eager_function(fun.init)(rng, x[0])
    initial_state = fun_init_state if initial_state is None else initial_state

    assert not isinstance(x, tuple)
    state_sequence = tf.scan(scan_f, x, initializer=initial_state)

    # TODO: find a more generic implementation
    output_sequence = state_sequence["eager_ewma"]["mean"]

    final_state = tree_map(lambda x: x[-1], state_sequence["eager_ewma"])
    return output_sequence, final_state
