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
"""Flax-based WAX-ML modules."""

from .apply_mask import ApplyMask
from .arma import ARMA
from .buffer import Buffer
from .counter import Counter
from .diff import Diff
from .ewma import EWMA
from .ewmcov import EWMCov
from .ewmvar import EWMVar
from .ffill import Ffill
from .fill_nan_inf import FillNanInf
from .has_changed import HasChanged
from .lag import Lag
from .mask_mean import MaskMean
from .mask_normalize import MaskNormalize
from .mask_std import MaskStd
from .ohlc import OHLC, OHLCData
from .online_optimizer import OnlineOptimizer, OptInfo
from .optax_optimizer import OptaxOptimizer
from .pct_change import PctChange
from .rolling_mean import RollingMean
from .vmap import VMap

__all__ = [
    "ApplyMask",
    "ARMA",
    "Buffer",
    "Counter",
    "Diff",
    "EWMA",
    "EWMCov",
    "EWMVar",
    "Ffill",
    "FillNanInf",
    "HasChanged",
    "Lag",
    "MaskMean",
    "MaskNormalize",
    "MaskStd",
    "OHLC",
    "OHLCData",
    "OnlineOptimizer",
    "OptaxOptimizer",
    "OptInfo",
    "PctChange",
    "RollingMean",
    "VMap",
]
