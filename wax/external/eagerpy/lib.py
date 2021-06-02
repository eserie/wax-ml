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
from .tensor import TensorType


def kl_div_with_logits(
    logits_p: TensorType, logits_q: TensorType, axis: int = -1, keepdims: bool = False
) -> TensorType:
    log_p = logits_p.log_softmax(axis=axis)
    log_q = logits_q.log_softmax(axis=axis)
    p = logits_p.softmax(axis=-1)
    return (p * (log_p - log_q)).sum(axis=axis, keepdims=keepdims)
