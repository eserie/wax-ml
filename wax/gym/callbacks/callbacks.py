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
b"""## Callbacks

We now develop a callbacks API in order to simplify as much as possible the local
training/simulation loop.

It permits to implement all complex stuffs in separated code that we call callbacks.
This can be to:
- record results
- compute additional metrics
- adding some monitoring tools: ProgressBar, ...
- control workflow: Stop iteration, ...

The design of the callback API bellow is highly inspired from the one used internally in the
library `dask` (see for ex `ProgressBar` callback).
"""

from contextlib import contextmanager
from typing import Set


class Callback:
    __slots__ = (
        "_cm",
        "_on_train_start",
        "_on_act",
        "_on_step",
        "_on_train_start",
        "_on_train_end",
    )
    """Base class for using the callback mechanism

    Create a callback with functions of the following signatures:

    >>> def _on_train_start(env, agent, obs):
    ...     pass
    >>> def on_act(env, agent, obs, action):
    ...     pass
    >>> def on_step(env, agent, gym_state):
    ...     pass
    >>> def _on_train_end(env, agent):
    ...     pass

    You may then construct a callback object with any number of them

    >>> cb = Callback(on_step=on_step) # doctest: +SKIP

    And use it either as a context manager over a compute/get call

    >>> with cb:            # doctest: +SKIP
    ...     gym_train(agent, env)     # doctest: +SKIP

    Or globally with the ``register`` method

    >>> cb.register()       # doctest: +SKIP
    >>> cb.unregister()     # doctest: +SKIP

    Alternatively subclass the ``Callback`` class with your own methods.

    >>> class PrintReward(Callback):      # doctest: +SKIP
    ...     def __init__(self): pass
    ...     def _on_step(env, agent, gym_state):
    ...         print(f"Reward = {gym_state.rw:.4g}")

    >>> with PrintReward():   # doctest: +SKIP
    ...     gym_train(agent, env)     # doctest: +SKIP
    """

    active: Set = set()

    def __init__(
        self, on_train_start=None, on_act=None, on_step=None, on_train_end=None
    ):
        if on_train_start:
            self._on_train_start = on_train_start
        if on_act:
            self._on_act = on_act
        if on_step:
            self._on_step = on_step
        if on_train_end:
            self._on_train_end = on_train_end

    @property
    def _callback(self):
        fields = ["_on_train_start", "_on_act", "_on_step", "_on_train_end"]
        return tuple(getattr(self, i, None) for i in fields)

    def __enter__(self):
        self._cm = add_callbacks(self)
        self._cm.__enter__()
        return self

    def __exit__(self, *args):
        self._cm.__exit__(*args)

    def register(self):
        Callback.active.add(self._callback)

    def unregister(self):
        Callback.active.remove(self._callback)


def unpack_callbacks(cbs):
    """Take an iterable of callbacks, return a list of each callback."""
    if cbs:
        return [[i for i in f if i] for f in zip(*cbs)]
    else:
        return [(), (), (), ()]


@contextmanager
def local_callbacks(callbacks=None):
    """Allows callbacks to work with nested schedulers.

    Callbacks will only be used by the first started scheduler they encounter.
    This means that only the outermost scheduler will use global callbacks."""
    global_callbacks = callbacks is None
    if global_callbacks:
        callbacks, Callback.active = Callback.active, set()
    try:
        yield callbacks or ()
    finally:
        if global_callbacks:
            Callback.active = callbacks


def normalize_callback(cb):
    """Normalizes a callback to a tuple"""
    if isinstance(cb, Callback):
        return cb._callback
    elif isinstance(cb, tuple):
        return cb
    else:
        raise TypeError("Callbacks must be either `Callback` or `tuple`")


class add_callbacks(object):
    """Context manager for callbacks.

    Takes several callbacks and applies them only in the enclosed context.
    Callbacks can either be represented as a ``Callback`` object, or as a tuple
    of length 4.

    Examples
    --------
    >>> def pretask(key, dsk, state):
    ...     print("Now running {0}").format(key)
    >>> callbacks = (None, pretask, None, None)
    >>> with add_callbacks(callbacks):    # doctest: +SKIP
    ...     res.compute()
    """

    def __init__(self, *callbacks):
        self.callbacks = [normalize_callback(c) for c in callbacks]
        Callback.active.update(self.callbacks)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        for c in self.callbacks:
            Callback.active.discard(c)
