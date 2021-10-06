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

import haiku as hk
import jax.numpy as jnp
import pytest
from haiku._src.data_structures import FlatMapping
from jax.config import config
from jax.numpy import array as DeviceArray
from jax.numpy import int32, uint32

from wax.external.eagerpy import convert_to_tensors
from wax.modules.gym_feedback import GymFeedback
from wax.testing import assert_tree_all_close
from wax.transform import (
    BatchState,
    transform_batch_with_state,
    transform_batch_with_state_static,
)
from wax.unroll import data_unroll, gym_static_unroll, unroll

from .counter import Counter
from .gym_feedback import GymOutput

GymState = GymFeedback.GymState


def raw_observations():
    for i in range(10):
        yield i, {"some_information_about_data"}


class Agent(hk.Module):
    def __call__(self, obs):
        action = obs + 1
        return {"structured_action": action}, {"agent_info": Counter()()}


class Env(hk.Module):
    def __call__(self, action, raw_obs):
        obs = raw_obs + action["structured_action"]
        rw = action["structured_action"] * obs
        return rw, obs, {"env_info": Counter()()}


def test_gym_module_gym_static_unroll():
    config.update("jax_enable_x64", False)

    @hk.transform_with_state
    def gym_fun(raw_obs):
        return GymFeedback(Agent(), Env(), return_obs=True, return_action=True)(raw_obs)

    raw_generator = iter(raw_observations())
    seq = hk.PRNGSequence(42)
    (gym_output, gym_info), state = gym_static_unroll(
        gym_fun, None, None, seq, True, raw_generator
    )

    ref_gym_output = GymOutput(return_obs=True, return_action=True).format(
        reward=DeviceArray([2, 15, 54, 140, 300, 567, 980, 1584, 2430], dtype=int32),
        obs=DeviceArray([2, 5, 9, 14, 20, 27, 35, 44, 54], dtype=int32),
        action={
            "structured_action": DeviceArray(
                [3, 6, 10, 15, 21, 28, 36, 45, 55], dtype=int32
            )
        },
    )

    ref_state = FlatMapping(
        {
            "gym_feedback": FlatMapping(
                {"action": GymState(action={"structured_action": 55})}
            ),
            "agent/counter": FlatMapping({"count": DeviceArray(9, dtype=uint32)}),
            "env/counter": FlatMapping({"count": DeviceArray(9, dtype=uint32)}),
        }
    )
    assert_tree_all_close(state, ref_state)
    assert_tree_all_close(gym_output, ref_gym_output)


def test_gym_module_dynamic_unroll():
    config.update("jax_enable_x64", False)

    @hk.transform_with_state
    def gym_fun(raw_obs):
        return GymFeedback(Agent(), Env())(raw_obs)

    xs = data_unroll(iter(raw_observations()))
    rng = next(hk.PRNGSequence(42))
    (gym_output, gym_info), final_state = unroll(
        gym_fun, return_final_state=True, rng=rng, skip_first=True
    )(xs)

    # reference outputs

    ref_final_state = FlatMapping(
        {
            "agent/counter": FlatMapping({"count": DeviceArray(9, dtype=uint32)}),
            "env/counter": FlatMapping({"count": DeviceArray(9, dtype=uint32)}),
            "gym_feedback": FlatMapping(
                {
                    "action": GymState(
                        action={"structured_action": DeviceArray(55, dtype=int32)}
                    ),
                }
            ),
        }
    )

    ref_output_sequence = GymOutput().format(
        reward=DeviceArray([2, 15, 54, 140, 300, 567, 980, 1584, 2430], dtype=int32),
    )

    assert_tree_all_close(final_state, ref_final_state)
    assert_tree_all_close(gym_output, ref_output_sequence)


@pytest.mark.parametrize("static", [True, False])
def test_gym_module_transform_batch_with_state(static):
    config.update("jax_enable_x64", False)

    @hk.transform_with_state
    def gym_fun(raw_obs):
        return GymFeedback(Agent(), Env())(raw_obs)

    xs = data_unroll(iter(raw_observations()))

    rng = next(hk.PRNGSequence(42))
    if static:
        xs, rng = convert_to_tensors((xs, rng), tensor_type="jax")
        batch_fun = transform_batch_with_state_static(gym_fun, skip_first=True)
    else:
        batch_fun = transform_batch_with_state(gym_fun, skip_first=True)
    batch_params, batch_state = batch_fun.init(rng, xs)
    print("first action: ", batch_state.fun_state)

    (output_sequence, info), batch_state = batch_fun.apply(
        batch_params, batch_state, rng, xs
    )

    assert batch_state._fields == ("fun_params", "fun_state", "rng_key")
    assert output_sequence._fields == ("reward",)

    assert (
        output_sequence.reward
        == jnp.array([2, 15, 54, 140, 300, 567, 980, 1584, 2430], dtype=jnp.int32)
    ).all()

    ref_output_sequence = GymOutput().format(
        reward=DeviceArray([2, 15, 54, 140, 300, 567, 980, 1584, 2430], dtype=int32),
    )
    ref_batch_state = BatchState(
        fun_params=FlatMapping({}),
        fun_state=FlatMapping(
            {
                "gym_feedback": FlatMapping(
                    {
                        "action": GymState(
                            action={"structured_action": DeviceArray(55, dtype=int32)}
                        ),
                    }
                ),
                "agent/counter": FlatMapping({"count": DeviceArray([9], dtype=uint32)}),
                "env/counter": FlatMapping({"count": DeviceArray([9], dtype=uint32)}),
            }
        ),
        rng_key=DeviceArray([255383827, 267815257], dtype=uint32),
    )

    assert_tree_all_close(batch_state, ref_batch_state)
    assert_tree_all_close(output_sequence, ref_output_sequence)
