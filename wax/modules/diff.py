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
"""Implement difference of values on sequential data."""
import haiku as hk

from wax.modules.buffer import Buffer


class Diff(hk.Module):
    """Implement difference of values on sequential data."""

    def __init__(self, periods: int = 1, name: str = None):
        """Initialize module.

        Args:
            periods : number of lags to use to compute the diff.
        """
        super().__init__(name=name)
        self.periods = periods
        assert periods == 1, "periods > 1 not implemented."

    def __call__(self, x):
        """Compute diff.

        Args:
            x: input data
        """
        buffer = Buffer(self.periods + 1)(x)
        diff = buffer[-1] - buffer[0]
        return diff
