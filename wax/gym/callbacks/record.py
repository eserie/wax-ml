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
"""TODO refactor this module"""

from typing import Any, Dict, List

import pandas as pd
import xarray as xr
from jax.tree_util import tree_flatten, tree_unflatten

import wax.external.eagerpy as ep
from wax.gym.callbacks import Callback
from wax.modules.gym_feedback import GymOutput
from wax.stream import DatasetSchema


class Record(Callback):
    """Record rewards while running gym_unroll.

    Parameters
    ----------

    schema: Dataset schema : contains coords and encoders.
    format_dims : nested data structure with specification of dims for dataarray formatting.

    return_obs
        if true, also record env observations.

    return_action
        if true, also record agent actions.

    Methods
    -------

    get_outputs

    """

    __slots__ = [
        "schema",
        "format_dims",
        "return_obs",
        "return_action",
        "GymOutput",
        "format_outputs",
        "_sequence",
        "_treedef",
        "_times",
        "_dates",
        "_env",
    ]

    def __init__(
        self,
        schema: DatasetSchema = None,
        format_dims: Any = None,
        return_obs: bool = False,
        return_action: bool = False,
        format_outputs: Any = None,
    ):
        self.schema = schema
        self.format_dims = format_dims
        self.return_obs = return_obs
        self.return_action = return_action
        self.format_outputs = format_outputs
        # state
        # same format as Gym module.
        self.GymOutput = GymOutput(self.return_obs, self.return_action)
        self._sequence = None
        self._treedef = None
        self._times: List = []
        self._dates: List = []
        self._env: Dict = {}

    def _on_step(self, env, agent, gym_state):
        if gym_state.done:
            return
        if gym_state.action is not None:

            outputs = self.GymOutput.format(
                gym_state.rw, gym_state.obs, gym_state.action
            )
            outputs, treedef = tree_flatten(outputs)
            if self._sequence is None:
                self._treedef = treedef
                self._sequence = tuple([] for _ in outputs)

            for seq_, output_ in zip(self._sequence, outputs):
                seq_.append(output_)

        self._env = env

    def get_outputs(self):
        """Get gym output sequence.

        Returns :
            unrolled output : reward recorded running gym_unroll.
                if return_obs is true, also return recorded env observations.
                if return_action is true, aslo return recorded agent actions.

        """
        sequence = ep.astensors(self._sequence)
        # sequence = self._sequence

        def _stack(x):
            if isinstance(x[0], (pd.DataFrame, pd.Series)):
                return pd.concat(x, axis=0)
            elif isinstance(x[0], (xr.Dataset, xr.DataArray)):
                return xr.concat(x, dim=x[0].dims[0])
            else:
                return ep.stack(x, axis=0)

        sequence = list(map(_stack, sequence))
        sequence = tree_unflatten(self._treedef, sequence)
        sequence = ep.as_raw_tensors(sequence)
        return sequence
