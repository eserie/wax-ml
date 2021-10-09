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
"""Unroll modules on data along first axis."""
import logging
import warnings
from collections import namedtuple
from typing import Any, Callable, NamedTuple, Union, cast

import haiku as hk
import jax
import jax.numpy as jnp
from haiku import TransformedWithState
from jax import tree_flatten, tree_unflatten
from jax._src.lax.control_flow import fori_loop
from jax.tree_util import tree_map
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


class UnrollTransformedWithState(NamedTuple):
    init: Callable
    apply: Callable


def unroll_transform_with_state(
    fun: Callable, skip_first: bool = False, dynamic: bool = True, pbar: bool = False
):
    """Transforms a function using Haiku modules into a pair of pure functions.
        which is unrolled on input arguments.

    Args:
        fun : callable or pair of pure functions (init, apply).
        skip_first : if true, first value of the sequence is not used in apply.
        dynamic : if true,  unroll using jax.lax.scan.
        pbar: if true, activate progress bar. Currently, it only works when dynamic=False.
    """
    if callable(fun):
        fun = hk.transform_with_state(fun)

    fun = cast(TransformedWithState, fun)

    def init(rng: jnp.ndarray, *args, **kwargs):
        xs = (args, kwargs)
        args_0, kwargs_0 = tree_map(lambda x: x[0], xs)
        params, state = fun.init(rng, *args_0, **kwargs_0)
        return params, state

    def dynamic_apply_fn(params: Any, state: Any, rng: jnp.ndarray, *args, **kwargs):
        xs = (args, kwargs)

        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)

        def scan_f(prev_state, inputs):
            args_step, kwargs_step = inputs
            outputs, next_state = fun.apply(
                params, prev_state, rng, *args_step, **kwargs_step
            )
            return next_state, outputs

        final_state, output_sequence = jax.lax.scan(scan_f, init=state, xs=xs)
        return output_sequence, final_state

    def static_apply_fn(params: Any, state: Any, rng: jnp.ndarray, *args, **kwargs):
        xs = (args, kwargs)

        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)

        def scan_f(prev_state, inputs):
            args_step, kwargs_step = inputs
            outputs, next_state = fun.apply(
                params, prev_state, rng, *args_step, **kwargs_step
            )
            return next_state, outputs

        final_state, output_sequence = static_scan(scan_f, init=state, xs=xs, pbar=pbar)
        return output_sequence, final_state

    if dynamic:
        return UnrollTransformedWithState(init, dynamic_apply_fn)
    else:
        return UnrollTransformedWithState(init, static_apply_fn)


def unroll(
    fun: Union[Callable, hk.TransformedWithState, UnrollTransformedWithState],
    skip_first: bool = False,
    dynamic: bool = True,
    pbar: bool = False,
    return_final_state=False,
    rng=None,
    params=None,
    state=None,
):
    """Transforms a function using Haiku modules into a function
    which is unrolled on input arguments.

    Args:
        fun : callable or pair of pure functions (init, apply).
        skip_first : if true, first value of the sequence is not used in apply.
        dynamic : if true,  unroll using jax.lax.scan.
        pbar: if true, activate progress bar. Currently, it only works when dynamic=False.
        return_final_state : if true, the returned function return unrolled outputs and final state.
        rng : if specified, used as rng key for the init and apply functions.
        params : if specified, used as params for the apply function.
        state : if sepecified, used as initial state for the apply function.

    Returns:
        apply_fn: wrapped function.

    """
    if not isinstance(fun, UnrollTransformedWithState):
        fun = unroll_transform_with_state(
            fun, skip_first=skip_first, dynamic=dynamic, pbar=pbar
        )

    def apply_fn(*args, **kwargs):
        fun_init_params, fun_init_state = fun.init(rng, *args, **kwargs)
        init_params = fun_init_params if params is None else params
        init_state = fun_init_state if state is None else state
        output, final_state = fun.apply(init_params, init_state, rng, *args, **kwargs)
        if return_final_state:
            return output, final_state
        else:
            return output

    return apply_fn


def init_params_state(
    fun: hk.TransformedWithState,
    rng: jnp.ndarray,
    *args,
    **kwargs,
):
    """Call init of a TransformedUnrollWithState pair."""
    warnings.warn(
        "Deprecated function init_params_state. Use unroll_transform_with_state instead."
        "This function may be removed in the near future.",
        DeprecationWarning,
        2,
    )

    return unroll_transform_with_state(fun).init(rng, *args, **kwargs)


def dynamic_unroll(
    fun: Union[Callable, hk.TransformedWithState],
    params: Any,
    state: Any,
    rng: jnp.ndarray,
    skip_first: bool = False,
    *args,
    **kwargs,
):
    """Unroll a TransformedWithState function using jax.lax.scan.

    Args:
        fun : callable or pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        rng: random number generator key.
        skip_first : if true, first value of the sequence is not used in apply.
        args, kwargs : Nested data structures with sequences as leaves passed to init and apply
            of the TransformedWithState pair.
    """
    warnings.warn(
        "Deprecated function dynamic_unroll. Use unroll or unroll_transform_with_state instead."
        "This function may be removed in the near future.",
        DeprecationWarning,
        2,
    )

    if callable(fun):
        fun = hk.transform_with_state(fun)

    fun = unroll_transform_with_state(fun, skip_first, dynamic=True)

    fun_init_params, fun_init_state = fun.init(rng, *args, **kwargs)
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state

    return fun.apply(params, state, rng, *args, **kwargs)


def static_unroll(
    fun: hk.TransformedWithState,
    params: Any,
    state: Any,
    rng: jnp.ndarray,
    skip_first: bool = False,
    *args,
    pbar: bool = False,
    **kwargs,
):
    """Unroll a TransformedWithState function using static_scan.

    Args:
        fun : pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        rng: random number generator key.
        skip_first : if true, first value of the sequence is not used in apply.
        args, kwargs : Nested datastructure with sequences as leaves passed to init and apply
            of the TransformedWithState pair.
        pbar : if true, activate progress bar.
    """
    warnings.warn(
        "Deprecated function static_unroll. Use unroll or unroll_transform_with_state instead."
        "This function may be removed in the near future.",
        DeprecationWarning,
        2,
    )

    if callable(fun):
        fun = hk.transform_with_state(fun)

    fun = unroll_transform_with_state(fun, skip_first, dynamic=False, pbar=pbar)

    fun_init_params, fun_init_state = fun.init(rng, *args, **kwargs)
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state

    return fun.apply(params, state, rng, *args, **kwargs)


def iter_first_axis(xs, pbar=False):
    """Iterate over first axis of a nested data structure."""
    xs_flat, treedef = tree_flatten(xs)
    T = len(xs_flat[0])
    if pbar:
        dates = tqdm(list(range(T)), desc="iter_first_axis (static_unroll)")
    else:
        dates = range(T)
    for t in dates:
        x_flat = map(lambda x: x[t], xs_flat)
        yield tree_unflatten(treedef, x_flat)


def static_scan(scan_f, init, xs=None, length=None, pbar=False):
    """Scan a function over leading array axes while carrying along state.

    Args:
        scan_f : function to apply
        init: initial value for first argument of scan_f
        xs : sequence of second argument of scan_f
        length: lenght of the output sequence.
        pbar: if true, activate a progress bar.

    Note:  Python implementation of jax.lax.scan
    See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax-lax-scan
    """
    if xs is None:
        xs = [None] * length
    carry = init
    ys = None
    for x in iter_first_axis(xs, pbar=pbar):
        carry, y = scan_f(carry, x)

        y, treedef = tree_flatten(y)
        if ys is None:
            ys = tuple([] for _ in y)
        for ys_, y_ in zip(ys, y):
            ys_.append(y_)

    ys = map(jnp.stack, ys)
    ys = tree_unflatten(treedef, ys)
    return carry, ys


# def static_scan_generator(scan_f, init, xs, length=None):
#     """Scan a function over leading array axes while carrying along state.
#
#     Python implementation of jax.lax.scan
#
#     See https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax-lax-scan
#     """
#     if xs is None:
#         xs = [None] * length
#     carry = init
#     for x in iter_first_axis(xs):
#         carry, y = scan_f(carry, x)
#         yield carry, y
#
#
# def static_unroll_generator(
#     fun, params, initial_state, key, xs, skip_first=False, pbar=True
# ):
#     """Generator version of static_unroll"""
#     # inline static_unroll
#     def scan_f(prev_state, inputs):
#         outputs, next_state = fun.apply(fun_params, prev_state, key, inputs)
#         return next_state, outputs
#
#     # init
#     x0 = tree_map(lambda x: x[0], xs)
#     fun_params, fun_init_state = fun.init(key, x0)
#     init_state = fun_init_state if initial_state is None else initial_state
#     # apply
#     if skip_first:
#         xs = tree_map(lambda x: x[1:], xs)
#     for carry, y in static_scan_generator(scan_f, init=init_state, xs=xs):
#         yield carry, y


def dynamic_unroll_fori_loop(
    fun: hk.TransformedWithState,
    params: Any,
    state: Any,
    rng: jnp.ndarray,
    skip_first: bool = False,
    *args,
    **kwargs,
):
    """Unroll a TransformedWithState function using fori_loop.

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

    fun_init_params, fun_init_state = init_params_state(fun, rng, *args, **kwargs)
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state
    xs = (args, kwargs)

    if skip_first:
        xs = tree_map(lambda x: x[1:], xs)

    # get template output
    args_0, kwargs_0 = tree_map(lambda x: x[0], xs)
    output_template, _ = fun.apply(params, state, rng, *args_0, **kwargs_0)

    T = len(tree_flatten(xs)[0][0])
    output_sequence = tree_map(
        lambda x: jnp.full((T,) + x.shape, jnp.nan, x.dtype), output_template
    )

    # run loop
    loop_state = LoopState(params, state, rng, xs, output_sequence)

    @jax.jit
    def body_fun(i, loop_state):
        params, prev_state, rng, inputs, output_sequence = loop_state
        args_i, kwargs_i = tree_map(lambda x: x[i], inputs)
        outputs, next_state = fun.apply(params, prev_state, rng, *args_i, **kwargs_i)
        output_sequence = jax.ops.index_update(output_sequence, i, outputs)
        return LoopState(
            params,
            next_state,
            rng,
            inputs,
            output_sequence,
        )

    loop_state = fori_loop(0, T, body_fun, loop_state)

    # format results
    final_state = loop_state.state
    output_sequence = loop_state.output_sequence
    return output_sequence, final_state


def data_unroll(data_gen, pbar=True):
    xs = None
    if pbar:
        data_gen = tqdm(data_gen, desc="data_unroll")
    for obs, info in data_gen:
        x = obs
        x, treedef = tree_flatten(x)
        if xs is None:
            # initialize xs
            xs = tuple([] for _ in x)
        for xs_, x_ in zip(xs, x):
            xs_.append(x_)
    xs = map(lambda x: jnp.stack(x, axis=0), xs)
    xs = tree_unflatten(treedef, xs)
    return xs


# def gym_dynamic_unroll(gym_fun, seq, data_generator, skip_first=True):
#     """Unroll a function over a data generator and random number generator."""
#     xs = data_unroll(data_generator)
#     output, gym_state = dynamic_unroll(gym_fun, xs, next(seq), skip_first=skip_first)
#     return output, gym_state


def gym_static_unroll(
    gym_fun, params, state, seq, skip_first, data_generator, pbar=False
):
    """Unroll a function over a data generator and random number generator.
    Args:
        gym_fun : pair of pure functions (init, apply).
        params: parameters for the function.
        state : state for the function.
        seq: generator of random number keys.
        skip_first : if true, first value of the sequence is not used in apply.
        data_generator : generator for gym_fun arguments.
    """
    obs, info = next(data_generator)
    fun_init_params, fun_init_state = gym_fun.init(next(seq), obs)
    params = fun_init_params if params is None else params
    state = fun_init_state if state is None else state

    ys = None
    if not skip_first:
        y, state = gym_fun.apply(params, state, next(seq), obs)
        y, treedef = tree_flatten(y)
        if ys is None:
            ys = tuple([] for _ in y)

    if pbar:
        data_generator = tqdm(data_generator, desc="gym_static_unroll")
    for obs, info in data_generator:
        y, state = gym_fun.apply(params, state, next(seq), obs)
        y, treedef = tree_flatten(y)
        if ys is None:
            ys = tuple([] for _ in y)
        for ys_, y_ in zip(ys, y):
            ys_.append(y_)

    ys = map(jnp.stack, ys)
    ys = tree_unflatten(treedef, ys)
    return ys, state
