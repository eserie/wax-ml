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
"""Apply a module when an event occur otherwise return last computed output.

Part of this module may be integrated in Haiku.
See: https://github.com/deepmind/dm-haiku/issues/126
"""
from typing import Mapping, Set

import haiku as hk
import jax.numpy as jnp
from haiku._src import base
from jax.tree_util import tree_map


def set_params_or_state_dict(
    module_name: str,
    submodules: Set[str],
    which: str,
    state_dict: Mapping[str, jnp.ndarray],
):
    """Returns module parameters or state for the given module or submodules."""
    assert which in ("params", "state")
    frame = base.current_frame()
    for their_module_name, bundle in getattr(frame, which).items():
        if (
            their_module_name == module_name
            or their_module_name.startswith(module_name + "/")
            or their_module_name in submodules
        ):
            for name, value in bundle.items():
                fq_name = their_module_name + "/" + name
                if which == "state":
                    value_dict = value._asdict()
                    value_dict["current"] = state_dict[fq_name]
                    bundle[name] = type(value)(**value_dict)
                else:
                    bundle[name] = state_dict[fq_name]


def set_state_from_dict(self, next_state_dict):
    """Set state keyed by name for this module and submodules."""
    if not base.frame_stack:
        raise ValueError(
            "`module.set_state_from_dict()` must be used as part of an `hk.transform`."
        )
    set_params_or_state_dict(
        self.module_name,
        self._submodules,
        "state",
        next_state_dict,
    )


hk.Module.set_state_from_dict = set_state_from_dict


class UpdateOnEvent(hk.Module):
    """Apply a module when an event occur otherwise return last computed output.

    If the module has state management, then it will be ask to delegate the state
    management to this module.
    """

    def __init__(self, module, initial_output_value=jnp.nan, name=None):
        super().__init__(name=name)
        self.module = module
        self.initial_output_value = initial_output_value

    def __call__(self, on_event, input):
        # state_dict = self.state_dict()

        prev_state_dict = self.module.state_dict()

        output = self.module(input)

        prev_output = hk.get_state(
            "prev_output",
            shape=[],
            init=lambda *_: tree_map(
                lambda x: jnp.full(x.shape, self.initial_output_value, dtype=x.dtype),
                output,
            ),
        )

        next_state_dict = self.module.state_dict() if prev_state_dict else {}

        def true_fun(operand):
            output, next_state, _, _ = operand
            return output, next_state

        def false_fun(operand):
            _, _, prev_output, prev_state = operand
            return prev_output, prev_state

        operand = (output, next_state_dict, prev_output, prev_state_dict)
        output, next_state_dict = hk.cond(
            pred=on_event,
            true_operand=operand,
            true_fun=true_fun,
            false_operand=operand,
            false_fun=false_fun,
        )

        if next_state_dict:
            self.module.set_state_from_dict(next_state_dict)

        hk.set_state("prev_output", output)
        return output
