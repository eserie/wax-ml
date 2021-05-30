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

from wax.gym.callbacks.callbacks import Callback, normalize_callback, unpack_callbacks


def test_callback():
    cb = Callback()
    assert isinstance(cb, Callback)

    cb = Callback(on_train_start=True)
    assert isinstance(cb, Callback)

    cb = Callback(on_act=True)
    assert isinstance(cb, Callback)

    cb = Callback(on_step=True)
    assert isinstance(cb, Callback)

    cb = Callback(on_train_end=True)
    assert isinstance(cb, Callback)

    cb.register()
    assert isinstance(cb, Callback)

    cb.unregister()
    assert isinstance(cb, Callback)
    _, _, _, _ = unpack_callbacks(None)
    ts, a, s, te = unpack_callbacks([[None, None, None, cb]])
    assert te[0] is cb
    with pytest.raises(TypeError):
        normalize_callback(None)
    ts, a, s, te = normalize_callback(cb)
    assert te is True
    cbn = normalize_callback((cb,))
    assert cbn == (cb,)
