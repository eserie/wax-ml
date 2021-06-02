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
from os.path import dirname as _dirname
from os.path import join as _join

from . import norms  # noqa: F401,E402
from . import types  # noqa: F401,E402
from . import utils  # noqa: F401,E402
from ._index import index  # noqa: F401,E402
from .astensor import as_raw_tensor  # noqa: F401,E402
from .astensor import as_raw_tensors  # noqa: F401,E402
from .astensor import astensor  # noqa: F401,E402
from .astensor import astensor_  # noqa: F401,E402
from .astensor import astensors  # noqa: F401,E402
from .astensor import astensors_  # noqa: F401,E402
from .astensor import eager_function  # noqa: F401,E402
from .convert import convert_to_tensor  # noqa: F401,E402
from .convert import convert_to_tensors  # noqa: F401,E402
from .framework import *  # noqa: F401,E402,F403
from .lib import *  # noqa: F401,E402,F403
from .modules import jax  # noqa: F401,E402
from .modules import numpy  # noqa: F401,E402
from .modules import tensorflow  # noqa: F401,E402
from .modules import torch  # noqa: F401,E402
from .tensor import JAXTensor  # noqa: F401,E402
from .tensor import NumPyTensor  # noqa: F401,E402
from .tensor import PyTorchTensor  # noqa: F401,E402
from .tensor import Tensor  # noqa: F401,E402
from .tensor import TensorFlowTensor  # noqa: F401,E402
from .tensor import TensorType  # noqa: F401,E402
from .tensor import istensor  # noqa: F401,E402

with open(_join(_dirname(__file__), "VERSION")) as _f:
    __version__ = _f.read().strip()
