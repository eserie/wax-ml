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
from typing import Any, Optional

import pytest

import wax.external.eagerpy as ep


def pytest_addoption(parser: Any) -> None:
    parser.addoption("--backend")


@pytest.fixture(scope="session")
def dummy(request: Any) -> ep.Tensor:
    backend: Optional[str] = request.config.option.backend
    if backend is None:
        pytest.skip()
        assert False
    return ep.utils.get_dummy(backend)


@pytest.fixture(scope="session")
def t1(dummy: ep.Tensor) -> ep.Tensor:
    return ep.arange(dummy, 5).float32()


@pytest.fixture(scope="session")
def t1int(dummy: ep.Tensor) -> ep.Tensor:
    return ep.arange(dummy, 5)


@pytest.fixture(scope="session")
def t2(dummy: ep.Tensor) -> ep.Tensor:
    return ep.arange(dummy, 7, 17, 2).float32()


@pytest.fixture(scope="session")
def t2int(dummy: ep.Tensor) -> ep.Tensor:
    return ep.arange(dummy, 7, 17, 2)


@pytest.fixture(scope="session", params=["t1", "t2"])
def t(request: Any, t1: ep.Tensor, t2: ep.Tensor) -> ep.Tensor:
    return {"t1": t1, "t2": t2}[request.param]
