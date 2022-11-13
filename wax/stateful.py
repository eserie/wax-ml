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
import functools
from typing import Any, Callable, Optional

import haiku as hk
import haiku._src.base as base
import jax
from haiku._src.stateful import (
    InternalState,
    add_split_rng_error,
    difference,
    get_mapped_axis_size,
    internal_state,
    temporary_internal_state,
    update_internal_state,
)
from haiku.experimental import lift_with_state
from jax.tree_util import tree_map


def list_to_tuple(x):
    return tuple(x) if isinstance(x, list) else x


@add_split_rng_error
def vmap_lift_with_state(
    fun: Callable[..., Any],
    in_axes=0,
    out_axes=0,
    axis_name: Optional[str] = None,
    axis_size: Optional[int] = None,
    params_axes: Optional[int] = 0,
    state_axes: Optional[int] = 0,
    *,
    split_rng: bool,
) -> Callable[..., Any]:
    """Modified version of :func:`haiku.vmap` with added params_axes and state_axes arguments.

    See :func:`haiku.vmap` docstring for more details.

    Args:
      fun: See :func:`jax.vmap`.
      in_axes: See :func:`jax.vmap`.
      out_axes: See :func:`jax.vmap`.
      axis_name: See :func:`jax.vmap`.
      axis_size: See :func:`jax.vmap`.
      params_axes: Axis to map over parameters.
      state_axes: Axis to map over state.
      split_rng: Controls whether random key APIs in Haiku (e.g.
        :func:`next_rng_key`) return different (aka. the internal key is split
        before calling your mapped function) or the same (aka. the internal key
        is broadcast before calling your mapped fucntion) key. See the docstring
        for examples.

    Returns:
      See :func:`jax.vmap`.
    """

    if not jax.tree_util.tree_leaves(in_axes):
        raise ValueError(
            f"{fun.__name__} must have at least one non-None value in in_axes "
            "to use with `hk.vmap`."
        )
    rng_axes = 0 if split_rng else None
    haiku_state_axes = InternalState(params_axes, state_axes, rng_axes)
    in_axes = list_to_tuple(in_axes), haiku_state_axes
    out_axes = out_axes, haiku_state_axes

    @functools.wraps(fun)
    def pure_fun(args, state_in):
        if split_rng:
            # NOTE: In the case of split_rng we recieve an RNG key (rather than the
            # internal state of a PRNGSequence) so we need to construct that here.
            rng = base.PRNGSequence(state_in.rng).internal_state
            state_in = InternalState(state_in.params, state_in.state, rng)

        with temporary_internal_state(state_in), base.push_jax_trace_level():
            out = fun(*args)
            state_out = difference(state_in, internal_state())
            return out, state_out

    @functools.wraps(fun)
    def mapped_fun(*args):
        base.assert_context("vmap")

        mapped_pure_fun = jax.vmap(
            pure_fun,
            in_axes=in_axes,
            out_axes=out_axes,
            axis_name=axis_name,
            axis_size=axis_size,
        )
        state = internal_state()

        if split_rng:
            # Need to take a new key and split.
            num = get_mapped_axis_size(args, in_axes[0])
            rng = base.next_rng_keys(num)
            state = internal_state()  # Needed since we mutated internal RNG.
            saved_rng = state.rng
            state = InternalState(state.params, state.state, rng)

        try:
            out, state = mapped_pure_fun(args, state)
        except ValueError as err:
            if split_rng and not base.params_frozen() and "out_axes" in str(err):
                # TODO(lenamartens): add error for state too.
                raise ValueError(
                    "hk.vmap does not support setting split_rng to True "
                    "during initialization because it assumes parameters "
                    "are always shared along the mapped dimension. "
                    "Consider switching the value of `split_rng` to False "
                    "during initialization through "
                    "`split_rng=(not hk.running_init())`."
                ) from err
            else:
                raise err

        if split_rng:
            state = InternalState(state.params, state.state, saved_rng)

        update_internal_state(state)

        return out

    return mapped_fun


def unroll_lift_with_state(fn: Callable, skip_first=False, split_rng=False):
    def apply_fn(*args, **kwargs):
        init, apply = hk.transform_with_state(fn)
        params_and_state_fn, updater = lift_with_state(init, name="f_lift")

        xs = (args, kwargs)
        args_0, kwargs_0 = tree_map(lambda x: x[0], xs)
        init_rng = hk.next_rng_key() if hk.running_init() else None
        init_rng = None
        params, state = params_and_state_fn(init_rng, *args_0, **kwargs_0)

        def scan_f(state, inputs):
            args_step, kwargs_step = inputs
            rng = hk.maybe_next_rng_key() if split_rng else None
            outputs, state = apply(params, state, rng, *args_step, **kwargs_step)
            return state, outputs

        xs = (args, kwargs)
        if skip_first:
            xs = tree_map(lambda x: x[1:], xs)
        final_state, output_sequence = hk.scan(scan_f, init=state, xs=xs)
        updater.update(final_state)

        return output_sequence

    return apply_fn
