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
from typing import Callable

import haiku as hk
from haiku.experimental import lift_with_state
import jax
from jax.tree_util import tree_map

def vmap_lift_with_state(fun: Callable, split_rng=False, init_rng=False):
    def apply_fn(*args, **kwargs):

        tfun = hk.transform_with_state(fun)    
        n_batches = len(jax.tree_util.tree_leaves((args, kwargs))[0])
        if hk.running_init() and init_rng:
            rng = hk.next_rng_key()
            rng = jax.random.split(rng, num=n_batches)
        else:
            rng = None

        @jax.vmap
        def init_fn(rng, *args, **kwargs):
            return tfun.init(rng, *args, **kwargs)

        params_and_state_fn, updater = lift_with_state(
            init_fn, name="f_lift", allow_reuse=False
        )
        params, state = params_and_state_fn(rng, *args, **kwargs)

        if split_rng:
            rng = hk.next_rng_key()
            rng = jax.random.split(rng, num=n_batches)
        else:
            rng = None
        out, state = jax.vmap(apply_fn, in_axes=[(0,), (0,), (0 if split_rng else None)])(params, state, rng, *args, **kwargs)

        updater.update(state)
        return out

    return apply_fn


def unroll_lift_with_state(
    fn: Callable, skip_first=False, split_rng=False, init_rng=False
):
    def apply_fn(*args, **kwargs):
        tfn = hk.transform_with_state(fn)
        params_and_state_fn, updater = lift_with_state(tfn.init, name="unroll_lift")

        def init(xs):
            args_0, kwargs_0 = tree_map(lambda x: x[0], xs)
            rng = hk.next_rng_key() if (hk.running_init() and init_rng) else None
            params, state = params_and_state_fn(rng, *args_0, **kwargs_0)
            return params, state

        def scan_f(state, inputs):
            args_step, kwargs_step = inputs
            rng = hk.maybe_next_rng_key() if split_rng else None
            outputs, state = tfn.apply(params, state, rng, *args_step, **kwargs_step)
            return state, outputs

        xs = (args, kwargs)
        params, state = init(xs)
        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)
        final_state, output_sequence = hk.scan(scan_f, init=state, xs=xs)
        updater.update(final_state)

        return output_sequence

    return apply_fn
