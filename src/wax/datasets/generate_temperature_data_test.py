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
import numpy as onp

from wax.datasets.generate_temperature_data import generate_times_data


def test_generate_timedata():
    times_data = generate_times_data()
    assert times_data.dtype == onp.dtype("<M8[ns]")
    assert times_data.shape == (124, 53, 25)
