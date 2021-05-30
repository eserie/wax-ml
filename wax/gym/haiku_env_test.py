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
from functools import partial

import haiku as hk
import numpy as np
import numpy as onp
import pandas as pd
import xarray as xr

from wax.gym.callbacks.progressbar import ProgressBar
from wax.gym.callbacks.record import Record
from wax.gym.callbacks.stop import Stop
from wax.gym.haiku_agent import HaikuAgent
from wax.gym.haiku_env import HaikuEnv


def gym_unroll(env, agent):
    obs = env.reset()
    agent.reset()
    while True:
        action = agent(obs)
        obs, rw, done, info = env.step(action)
        print(action, obs, rw, done, info)
        if done:
            break
    return action, obs, rw, done, info


def raw_observations():
    for i in range(10):
        yield i, {"some_information_about_data"}


def raw_observations_array():
    for i in range(10):
        yield onp.full((1, 1), i), {"some_information_about_data"}


def raw_observations_dataframe():
    for i in range(10):
        yield pd.DataFrame(onp.full((1, 1), i)), {"some_information_about_data"}


def raw_observations_dataarray():
    for i in range(10):
        yield xr.DataArray(onp.full((1, 1), i)), {"some_information_about_data"}


class Agent(hk.Module):
    def __call__(self, obs):
        action = obs + 1
        return {"structured_action": action}


class Env(hk.Module):
    def __call__(self, action, raw_obs):
        obs = raw_obs + action["structured_action"]
        rw = action["structured_action"] * obs
        return rw, obs


def test_haiku_agent_and_env():
    data_generator = iter(raw_observations())

    @partial(HaikuEnv, data_generator)
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @HaikuAgent
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    action, obs, rw, done, info = gym_unroll(env, agent)
    assert action["structured_action"] == 55


def test_haiku_agent_and_env2():
    @partial(HaikuEnv, iter(raw_observations()), name="Env")
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @partial(HaikuAgent, name="Agent")
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    action, obs, rw, done, info = gym_unroll(env, agent)

    assert action["structured_action"] == 55


def test_unroll():
    @partial(HaikuEnv, iter(raw_observations()), name="Env")
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @partial(HaikuAgent, name="Agent")
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    with Record() as rec:
        state = agent.unroll(env)

    assert state.action["structured_action"] == 55
    outputs = rec.get_outputs()
    assert isinstance(outputs.reward, onp.ndarray)
    assert (
        outputs.reward == onp.array([2, 15, 54, 140, 300, 567, 980, 1584, 2430])
    ).all()


def test_unroll_array():
    @partial(HaikuEnv, iter(raw_observations_array()), name="Env")
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @partial(HaikuAgent, name="Agent")
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    with Record() as rec, ProgressBar():

        state = agent.unroll(env)

    assert isinstance(state.action["structured_action"], np.ndarray)
    assert state.action["structured_action"] == 55

    outputs = rec.get_outputs()
    assert isinstance(outputs.reward, np.ndarray)
    assert outputs.reward.tolist() == (
        [[[2]], [[15]], [[54]], [[140]], [[300]], [[567]], [[980]], [[1584]], [[2430]]]
    )


def test_unroll_dataframe():
    @partial(HaikuEnv, iter(raw_observations_dataframe()), name="Env")
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @partial(HaikuAgent, name="Agent")
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    with Record() as rec, ProgressBar():

        state = agent.unroll(env)

    assert isinstance(state.action["structured_action"], pd.DataFrame)
    assert state.action["structured_action"].values == 55

    outputs = rec.get_outputs()
    assert isinstance(outputs.reward, pd.DataFrame)
    assert outputs.reward.values.tolist() == (
        [[2], [15], [54], [140], [300], [567], [980], [1584], [2430]]
    )


def test_unroll_dataframe_stop_callback():
    @partial(HaikuEnv, iter(raw_observations_dataframe()), name="Env")
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @partial(HaikuAgent, name="Agent")
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    with Record() as rec, ProgressBar(), Stop(3):

        state = agent.unroll(env)

    assert isinstance(state.action["structured_action"], pd.DataFrame)
    assert state.action["structured_action"].values == 6

    outputs = rec.get_outputs()
    assert isinstance(outputs.reward, pd.DataFrame)
    assert outputs.reward.values.tolist() == ([[2], [15], [54]])


def test_unroll_dataarray_stop_callback():
    @partial(HaikuEnv, iter(raw_observations_dataarray()), name="Env")
    @hk.transform_with_state
    def env(action, raw_obs):
        return Env()(action, raw_obs)

    @partial(HaikuAgent, name="Agent")
    @hk.transform_with_state
    def agent(obs):
        return Agent()(obs)

    with Record() as rec, ProgressBar(), Stop(3):

        state = agent.unroll(env)

    assert isinstance(state.action["structured_action"], xr.DataArray)
    assert state.action["structured_action"].values == 6

    outputs = rec.get_outputs()
    assert isinstance(outputs.reward, xr.DataArray)
    assert outputs.reward.values.tolist() == ([[2], [15], [54]])
