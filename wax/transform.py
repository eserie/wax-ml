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
"""Transformation functions to work on batches of data."""

from typing import Any, Callable, NamedTuple

import jax
from jax import tree_map

from wax.unroll import static_scan


class TransformedBatch(NamedTuple):
    init: Callable
    apply: Callable


class BatchState(NamedTuple):
    fun_params: Any
    fun_state: Any
    rng_key: Any


class BatchParams(NamedTuple):
    ...


def transform_batch_with_state(fun, skip_first=False):
    """Transforms a pair of pure functions into another pair of pure functions
    which run on a batch.
    """

    def scan_f(prev_state, inputs):
        fun_params, fun_state, key = prev_state
        outputs, fun_state = fun.apply(fun_params, fun_state, key, inputs)
        return BatchState(fun_params, fun_state, key), outputs

    def init(key, xs):
        x0 = tree_map(lambda x: x[0], xs)
        fun_params, fun_state = fun.init(key, x0)
        return BatchParams(), BatchState(fun_params, fun_state, key)

    def apply(params, state, key, xs):
        # params is not used ... (and is empty)
        # replace state rng_key by given rng_key
        # we drop the first observation because it has been used in initialization....
        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)
        final_state, output_sequence = jax.lax.scan(scan_f, init=state, xs=xs)
        assert isinstance(final_state, BatchState)
        return output_sequence, final_state

    return TransformedBatch(init, apply)


def transform_batch_with_state_static(fun, skip_first=False, pbar=True):
    """Transforms a pair of pure functions into another pair of pure functions
    which run on a batch.
    """

    def scan_f(prev_state, inputs):
        fun_params, fun_state, key = prev_state
        outputs, fun_state = fun.apply(fun_params, fun_state, key, inputs)
        return BatchState(fun_params, fun_state, key), outputs

    def init(key, xs):
        x0 = tree_map(lambda x: x[0], xs)
        fun_params, fun_state = fun.init(key, x0)
        return BatchParams(), BatchState(fun_params, fun_state, key)

    def apply(params, state, key, xs):
        # params is not used ... (and is empty)
        # replace state rng_key by given rng_key
        init_state = BatchState(state.fun_params, state.fun_state, key)
        # we drop the first observation because it has been used in initialization....
        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)
        final_state, output_sequence = static_scan(scan_f, init=init_state, xs=xs)
        return output_sequence, final_state

    return TransformedBatch(init, apply)
