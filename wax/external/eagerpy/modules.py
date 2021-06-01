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
import functools
import inspect
from importlib import import_module
from types import ModuleType
from typing import Any, Callable, Iterable

from .astensor import astensor


def wrap(f: Callable) -> Callable:
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = f(*args, **kwargs)
        try:
            result = astensor(result)
        except ValueError:
            pass
        return result

    return wrapper


class ModuleWrapper(ModuleType):
    """A wrapper for modules that delays the import until it is needed
    and wraps the output of functions as EagerPy tensors"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if self.__doc__ is None:
            self.__doc__ = f"EagerPy wrapper of the '{self.__name__}' module"

    def __dir__(self) -> Iterable[str]:
        # makes sure tab completion works
        return import_module(self.__name__).__dir__()

    def __getattr__(self, name: str) -> Any:
        attr = getattr(import_module(self.__name__), name)
        if callable(attr):
            attr = wrap(attr)
        elif inspect.ismodule(attr):
            attr = ModuleWrapper(attr.__name__)
        return attr


torch = ModuleWrapper("torch")
tensorflow = ModuleWrapper("tensorflow")
jax = ModuleWrapper("jax")
numpy = ModuleWrapper("numpy")
