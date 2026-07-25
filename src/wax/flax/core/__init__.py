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
"""Core Flax-based WAX-ML functionality."""

from .streaming_transforms import (
    ConditionalComputation,
    StreamingScan,
    StreamingTransform,
    streaming_scan,
    streaming_transform_with_state,
    update_on_event,
)
from .transform import FlaxTransformed, flax_transform_with_state
from .unroll import flax_unroll, flax_unroll_transform

__all__ = [
    "FlaxTransformed",
    "flax_transform_with_state",
    "flax_unroll",
    "flax_unroll_transform",
    "StreamingTransform",
    "streaming_transform_with_state",
    "update_on_event",
    "streaming_scan",
    "ConditionalComputation",
    "StreamingScan",
]
