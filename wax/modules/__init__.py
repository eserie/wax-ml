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


from wax.modules.apply_mask import ApplyMask
from wax.modules.arma import ARMA
from wax.modules.buffer import Buffer
from wax.modules.counter import Counter
from wax.modules.diff import Diff
from wax.modules.ewma import EWMA
from wax.modules.ewmcov import EWMCov
from wax.modules.ewmvar import EWMVar
from wax.modules.ffill import Ffill
from wax.modules.fill_nan_inf import FillNanInf
from wax.modules.gym_feedback import GymFeedback
from wax.modules.has_changed import HasChanged
from wax.modules.lag import Lag
from wax.modules.mask_mean import MaskMean
from wax.modules.mask_normalize import MaskNormalize
from wax.modules.mask_std import MaskStd
from wax.modules.ohlc import OHLC
from wax.modules.online_optimizer import OnlineOptimizer
from wax.modules.online_supervised_learner import OnlineSupervisedLearner
from wax.modules.pct_change import PctChange
from wax.modules.rolling_mean import RollingMean
from wax.modules.snarimax import SNARIMAX
from wax.modules.update_on_event import UpdateOnEvent
from wax.modules.update_params import UpdateParams
from wax.modules.vmap import VMap

__all__ = [
    "Buffer",
    "Diff",
    "EWMA",
    "EWMCov",
    "EWMVar",
    "GymFeedback",
    "HasChanged",
    "Lag",
    "OHLC",
    "Ffill",
    "PctChange",
    "RollingMean",
    "UpdateOnEvent",
    "FillNanInf",
    "ApplyMask",
    "MaskMean",
    "MaskStd",
    "MaskNormalize",
    "OnlineSupervisedLearner",
    "VMap",
    "UpdateParams",
    "ARMA",
    "SNARIMAX",
    "OnlineOptimizer",
    "Counter",
]
