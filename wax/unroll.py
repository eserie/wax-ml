# Copyright 2021 The Wax Authors
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
from collections import namedtuple

import jax
from jax import numpy as onp
from jax import tree_flatten, tree_unflatten
from jax._src.lax.control_flow import fori_loop
from jax.tree_util import tree_map
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def dynamic_unroll(fun, xs, key=None, initial_state=None, skip_first=False):
    """Implementation with jax.lax.scan."""

    def scan_f(prev_state, inputs):
        outputs, next_state = fun.apply(params, prev_state, key, inputs)
        return next_state, outputs

    x0 = tree_map(lambda x: x[0], xs)
    params, fun_init_state = fun.init(key, x0)
    initial_state = fun_init_state if initial_state is None else initial_state

    if skip_first:
        xs = tree_map(lambda x: x[1:], xs)
    final_state, output_sequence = jax.lax.scan(scan_f, init=initial_state, xs=xs)
    return output_sequence, final_state


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


def static_scan(scan_f, init, xs, length=None, pbar=False):
    """Scan a function over leading array axes while carrying along state.

    Python implementation of jax.lax.scan

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

    ys = map(onp.stack, ys)
    ys = tree_unflatten(treedef, ys)
    return carry, ys


def static_unroll(fun, xs, key=None, initial_state=None, skip_first=False, pbar=True):
    def scan_f(prev_state, inputs):
        outputs, next_state = fun.apply(fun_params, prev_state, key, inputs)
        return next_state, outputs

    # init
    x0 = tree_map(lambda x: x[0], xs)
    fun_params, fun_init_state = fun.init(key, x0)
    init_state = fun_init_state if initial_state is None else initial_state

    # apply
    if skip_first:
        xs = tree_map(lambda x: x[1:], xs)
    final_state, output_sequence = static_scan(
        scan_f, init=init_state, xs=xs, pbar=pbar
    )

    return output_sequence, final_state


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
#     fun, xs, key=None, initial_state=None, skip_first=False, pbar=True
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


def dynamic_unroll_fori_loop(fun, x, key=None, initial_state=None):
    LoopState = namedtuple("LoopState", "params, state, rng, x, output_sequence")

    @jax.jit
    def body_fun(i, loop_state):
        params, prev_state, rng, inputs, output_sequence = loop_state
        outputs, next_state = fun.apply(params, prev_state, rng, inputs[i])
        output_sequence = jax.ops.index_update(output_sequence, i, outputs)
        return LoopState(
            params,
            next_state,
            rng,
            inputs,
            output_sequence,
        )

    # Initialize loop
    params, fun_init_state = fun.init(key, x[0])
    initial_state = fun_init_state if initial_state is None else initial_state
    output_sequence = onp.full_like(x, onp.nan)
    loop_state = LoopState(params, initial_state, key, x, output_sequence)

    # run loop
    T = len(x)
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
    xs = map(lambda x: onp.stack(x, axis=0), xs)
    xs = tree_unflatten(treedef, xs)
    return xs


# def gym_dynamic_unroll(gym_fun, data_generator, seq, skip_first=True):
#     """Unroll a function over a data generator and random number generator."""
#     xs = data_unroll(data_generator)
#     output, gym_state = dynamic_unroll(gym_fun, xs, next(seq), skip_first=skip_first)
#     return output, gym_state


def gym_static_unroll(gym_fun, data_generator, seq, skip_first=True, pbar=True):
    """Unroll a function over a data generator and random number generator."""
    obs, info = next(data_generator)
    params, state = gym_fun.init(next(seq), obs)

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

    ys = map(onp.stack, ys)
    ys = tree_unflatten(treedef, ys)
    return ys, state
