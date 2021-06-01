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
import pytest

import wax.external.eagerpy as ep


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_kl_div_with_logits(dummy: ep.Tensor, axis: int) -> None:
    logits_p = logits_q = ep.arange(dummy, 12).float32().reshape((3, 4))
    assert (ep.kl_div_with_logits(logits_p, logits_q, axis=axis) == 0).all()
