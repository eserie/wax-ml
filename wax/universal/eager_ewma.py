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
"""Universal Exponential moving average module and unroll implementations.
"""
from collections import namedtuple
from typing import Any, Tuple, cast

import haiku as hk
import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.lax import fori_loop
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten
from tqdm.auto import tqdm

import wax.external.eagerpy as ep
from wax.external.eagerpy import eager_function


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
def static_unroll_universal(
    fun: hk.TransformedWithState,
    params: Any,
    state: Any,
    rng: jnp.ndarray,
    skip_first: bool = False,
    *args,
    pbar=True,
    compile=None,
    **kwargs,
):
    """Unroll a TransformedWithState function using static_scan implemented
    in an universal way.

    Args:
        fun : pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        rng: random number generator key.
        skip_first : if True, first value of the sequence is not used in apply.
        args, kwargs : Nested datastructure with sequences as leaves passed to init and apply
            of the TransformedWithState pair.
        pbar : if true, activate progress bar.
        compile : if true, compile the the appropriate backend.
    """
    fun = type(fun)(eager_function(fun.init), eager_function(fun.apply))

    # init
    xs = (args, kwargs)
    x0 = tree_map(lambda x: x[0], xs)
    fun_init_params, fun_init_state = fun.init(rng, *x0[0], **x0[1])
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state
    # apply
    if skip_first:
        xs = tree_map(lambda x: x[1:], xs)

    if compile == "jax":
        fun = type(fun)(fun.init, jax.jit(fun.apply))
    if compile == "tensorflow":
        fun = type(fun)(fun.init, tf.function(fun.apply))

    xs, treedef = tree_flatten(xs)
    T = len(xs[0])

    output_template, _ = fun.apply(params, state, rng, *x0[0], **x0[1])
    output_template, output_treedef = tree_flatten(output_template)
    output_sequence: Tuple = tuple(map(lambda x: [], output_template))

    steps = range(T)
    if pbar:
        steps = tqdm(list(steps))
    for i in steps:
        args_i, kwargs_i = tree_unflatten(
            treedef, tuple(map(lambda x: cast(Tuple, x)[i], xs))
        )
        outputs, state = fun.apply(params, state, rng, *args_i, **kwargs_i)
        outputs = tree_leaves(outputs)
        for os, o in zip(output_sequence, outputs):
            os.append(o)

    @eager_function
    def _stack(x):
        return ep.stack(x)

    output_sequence = tuple(map(lambda x: _stack(x), output_sequence))
    output_sequence = tree_unflatten(output_treedef, output_sequence)
    return output_sequence, state


@eager_function
def dynamic_unroll_fori_loop_universal(
    fun: hk.TransformedWithState,
    params: Any,
    state: Any,
    rng: jnp.ndarray,
    skip_first: bool = False,
    *args,
    **kwargs,
):
    """Unroll a TransformedWithState function using fori_loop in an universal way.

    Args:
        fun : pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        rng: random number generator key.
        skip_first : if true, first value of the sequence is not used in apply.
        args, kwargs : Nested datastructure with sequences as leaves passed to init and apply
            of the TransformedWithState pair.
    """

    LoopState = namedtuple("LoopState", "params, state, rng, x, output_sequence")
    fun = type(fun)(eager_function(fun.init), eager_function(fun.apply))

    # init
    xs = (args, kwargs)
    x0 = tree_map(lambda x: x[0], xs)
    fun_init_params, fun_init_state = fun.init(rng, *x0[0], **x0[1])
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state
    # apply
    if skip_first:
        xs = tree_map(lambda x: x[1:], xs)

    @jax.jit
    def body_fun(i, loop_state):
        params, prev_state, rng, inputs, output_sequence = loop_state
        args_i, kwargs_i = tree_map(lambda x: x[i], inputs)
        outputs, next_state = eager_function(fun.apply)(
            params, prev_state, rng, *args_i, **kwargs_i
        )
        output_sequence = ep.index_update(output_sequence, i, outputs)
        return LoopState(
            params,
            next_state,
            rng,
            inputs,
            output_sequence,
        )

    output_template, _ = fun.apply(params, state, rng, *x0[0], **x0[1])
    T = len(tree_flatten(xs)[0][0])

    @eager_function
    def _full_like_T_nan(x):
        x = ep.stack([x] * T)
        return ep.full_like(x, ep.nan)

    output_sequence = tree_map(lambda x: _full_like_T_nan(x), output_template)
    loop_state = LoopState(params, state, rng, xs, output_sequence)

    # run loop
    loop_state = fori_loop(0, T, body_fun, loop_state)

    # format results
    final_state = loop_state.state
    output_sequence = loop_state.output_sequence
    return output_sequence, final_state


@eager_function
def dynamic_unroll_universal(
    fun: hk.TransformedWithState,
    params: Any,
    state: Any,
    rng: jnp.ndarray,
    skip_first: bool = False,
    *args,
    **kwargs,
):
    """Unroll a TransformedWithState function using jax.lax.scan.

    Args:
        fun : pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        rng: random number generator key.
        skip_first : if true, first value of the sequence is not used in apply.
        args, kwargs : Nested datastructure with sequences as leaves passed to init and apply
            of the TransformedWithState pair.

    We can use scan because the output mean is in the state!

    See : https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html
    """
    fun = type(fun)(eager_function(fun.init), eager_function(fun.apply))

    xs = (args, kwargs)
    x0 = tree_map(lambda x: x[0], xs)
    fun_init_params, fun_init_state = fun.init(rng, *x0[0], **x0[1])
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state
    if skip_first:
        xs = tree_map(lambda x: x[1:], xs)

    def scan_f(prev_state, inputs):
        outputs, next_state = fun.apply(
            params, prev_state, rng, *inputs[0], **inputs[1]
        )
        return next_state, outputs

    final_state, output_sequence = jax.lax.scan(scan_f, init=state, xs=xs)
    return output_sequence, final_state


@tf.function
@tf.autograph.experimental.do_not_convert
def dynamic_unroll_tf(fun, params, state, rng, skip_first, x):
    """Unroll a TransformedWithState function using tensorflow
    scan function.

    Args:
        fun : pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        rng: random number generator key.
        skip_first : if true, first value of the sequence is not used in apply.
        x : Nested datastructure with sequences as leaves passed to init and apply
            of the TransformedWithState pair.
    """
    rng = rng if rng is not None else next(hk.PRNGSequence(42))

    def scan_f(prev_state, inputs):
        outputs, next_state = eager_function(fun.apply)(params, prev_state, rng, inputs)
        return next_state

    params, fun_init_state = eager_function(fun.init)(rng, x[0])
    state = fun_init_state if state is None else state

    assert not isinstance(x, tuple)
    state_sequence = tf.scan(scan_f, x, initializer=state)

    # TODO: find a more generic implementation
    output_sequence = state_sequence["eager_ewma"]["mean"]

    final_state = tree_map(lambda x: x[-1], state_sequence["eager_ewma"])
    return output_sequence, final_state
