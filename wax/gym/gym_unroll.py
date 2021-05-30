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
"""Unroll agent with a gym environment."""
from collections import namedtuple

from wax.gym.callbacks.callbacks import local_callbacks, unpack_callbacks


def gym_unroll(agent, env, obs=None, callbacks=None):
    """Unroll agent with a gym environment.

    Parameters
    ----------
    env
        Gym environment used to unroll the data and feed the agent.

    obs
        if passed, start the Gym unroll loop with this observation instead of
        the observation returned by 'env.reset()', else 'obs = env.reset()' is
        used as first observation.

    callbacks
        Additional callback to use

    Returns
    -------

    gym_state
        nested data structure
    """
    GymUnrollState = namedtuple("GymUnrollState", "rw, obs, action, done, info")

    with local_callbacks(callbacks) as callbacks:

        # unpack callbacks
        (
            on_train_start_cbs,
            on_act_cbs,
            on_step_cbs,
            on_train_end_cbs,
        ) = unpack_callbacks(callbacks)

        obs = env.reset() if obs is None else obs

        # call "on_train_start" callbacks
        for on_train_start in on_train_start_cbs:
            on_train_start(env, agent, obs)

        while True:

            # agent act
            action = agent(obs)

            # call "on_act" callbacks
            for on_act in on_act_cbs:
                on_act(env, agent, obs, action)

            # env step
            obs, rw, done, info = env.step(action)

            # set loop state
            gym_state = GymUnrollState(rw, obs, action, done, info)

            # call "on_step" callbacks
            for on_step in on_step_cbs:
                step_done = on_step(env, agent, gym_state)
                done = done or step_done

            if done:
                break

        # call "on_train_end" callbacks
        for on_train_end in on_train_end_cbs:
            on_train_end(env, agent)

        env.close()

    return gym_state
