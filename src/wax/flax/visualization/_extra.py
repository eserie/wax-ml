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
"""Shared wording for the `visualization` extra.

Every module in this package guards its backend imports so that
``import wax.flax.visualization`` succeeds on a plain ``wax-ml`` install. When a
guarded feature is then used, the error must say what to install -- these helpers
keep that single sentence identical everywhere.
"""

from __future__ import annotations

VISUALIZATION_EXTRA_HINT = "Install the visualization extra: pip install 'wax-ml[visualization]'"


def require(available: bool, feature: str, backends: str) -> None:
    """Raise an actionable `ImportError` when a backend-dependent feature is used.

    Args:
        available: The module-level ``HAS_*`` flag for the backend.
        feature: What the caller was trying to do, e.g. ``"streaming plots"``.
        backends: The missing backend(s), e.g. ``"Bokeh"``.

    Raises:
        ImportError: If `available` is false, naming `backends` and the extra.
    """
    if not available:
        raise ImportError(f"{backends} is required for {feature}. {VISUALIZATION_EXTRA_HINT}")
