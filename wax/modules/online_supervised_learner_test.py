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
"""We implement an online learning non-stationary linear regression problem.

We go there progressively by showing how a linear regression problem can be cast
into an online learning problem thanks to the `OnlineSupervisedLearner` module.

Then, in order to tackle a non-stationary linear regression problem (i.e. with a weight that can vary in time)
we reformulate the problem into a reinforcement learning problem that we implement with the `GymFeedBack` module of WAX-ML.

We then need to define an "agent" and an "environment" using simple functions or modules:
- The agent is responsible for learning the weights of its internal linear model.
- The environment is responsible for generating labels and evaluating the agent's reward metric.

We experiment with a non-stationary environment that returns the sign of the linear regression parameters at a given time step,
known only to the environment.

We will see that doing this is very simple with the WAX-ML tools and that the functional workflow it adopts
allows, each time we increase in complexity, to reuse the previously implemented transformations.


In this journey, we will use:
- Haiku basic linear module `hk.Linear`.
- Optax stochastic gradient descent optimizer: `sgd`.
- WAX-ML modules: `OnlineSupervisedLearner`, `Lag`, `GymFeedBack`
- WAX-ML helper functions: `dynamic_unroll`, `jit_init_apply`
"""
import haiku as hk
import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_map

from wax.compile import jit_init_apply
from wax.modules import GymFeedback, Lag, OnlineSupervisedLearner
from wax.unroll import dynamic_unroll


@jit_init_apply
@hk.transform_with_state
def linear_model(x):
    return hk.Linear(output_size=1, with_bias=False)(x)


def test_static_model():
    # First let's implement a simple linear regression
    # Let's generate a batch of data:

    seq = hk.PRNGSequence(42)
    T = 100
    N = 3
    X = jax.random.normal(next(seq), (T, N))
    w_true = jnp.ones(N)

    params, state = linear_model.init(next(seq), X[0])
    linear_model.apply(params, state, None, X[0])

    Y_pred, state = dynamic_unroll(linear_model, None, None, next(seq), False, X)

    assert Y_pred.shape == (T, 1)

    noise = jax.random.normal(next(seq), (T,))
    Y = X.dot(w_true) + noise
    mean_loss = ((Y - Y_pred) ** 2).sum(axis=1).mean()
    assert mean_loss > 0


def generate_many_observations(T=300, sigma=1.0e-2, rng=None):
    rng = jax.random.PRNGKey(42) if rng is None else rng
    X = jax.random.normal(rng, (T, 3))
    noise = sigma * jax.random.normal(rng, (T,))
    w_true = jnp.ones(3)
    noise = sigma * jax.random.normal(rng, (T,))
    Y = X.dot(w_true) + noise
    return (X, Y)


def test_online_model():
    # # Online model

    opt = optax.sgd(1e-3)

    @jax.jit
    def loss(y_pred, y):
        return jnp.mean(jnp.square(y_pred - y))

    @jit_init_apply
    @hk.transform_with_state
    def learner(x, y):
        return OnlineSupervisedLearner(linear_model, opt, loss)(x, y)

    seq = hk.PRNGSequence(42)

    # generate data
    T = 300
    X, Y = generate_many_observations(T)

    # dynamic unroll the learner
    x0, y0 = tree_map(lambda x: x[0], (X, Y))
    online_params, online_state = learner.init(next(seq), x0, y0)
    output, online_state = dynamic_unroll(
        learner, online_params, online_state, next(seq), False, X, Y
    )
    assert len(output["loss"]) == T
    assert len(output["params"]["linear"]["w"])


def linear_regression_agent(obs):
    x, y = obs

    opt = optax.sgd(1e-3)

    @jax.jit
    def loss(y_pred, y):
        return jnp.mean(jnp.square(y_pred - y))

    def learner(x, y):
        return OnlineSupervisedLearner(linear_model, opt, loss)(x, y)

    return learner(x, y)


def stationary_linear_regression_env(action, raw_obs):
    # Only the environment now the true value of the parameters
    w_true = -jnp.ones(3)

    # The environment has its proper loss definition
    @jax.jit
    def loss(y_pred, y):
        return jnp.mean(jnp.square(y_pred - y))

    # raw observation contains features and generative noise
    x, noise = raw_obs

    # generate targets
    y = x @ w_true + noise
    obs = (x, y)

    y_previous = Lag(1)(y)
    # evaluate the prediction made by the agent
    y_pred = action["y_pred"]
    reward = loss(y_pred, y_previous)

    return reward, obs


def generate_many_raw_observations(T=300, sigma=1.0e-2, rng=None):
    rng = jax.random.PRNGKey(42) if rng is None else rng
    X = jax.random.normal(rng, (T, 3))
    noise = sigma * jax.random.normal(rng, (T,))
    return (X, noise)


def test_online_recast_as_reinforcement_learning_pb():
    # # Online supervised learning recast as a reinforcement learning problem
    # obs = (x, y) are tuple observations.
    # raw_obs = (x, noise) consist in the feature and input noise.

    @hk.transform_with_state
    def gym_fun(raw_obs):
        return GymFeedback(
            linear_regression_agent,
            stationary_linear_regression_env,
            return_action=True,
        )(raw_obs)

    T = 300
    raw_observations = generate_many_raw_observations(T)
    rng = jax.random.PRNGKey(42)
    output_sequence, final_state = dynamic_unroll(
        gym_fun,
        None,
        None,
        rng,
        True,
        raw_observations,
    )

    assert len(output_sequence.reward) == T - 1
    assert len(output_sequence.action["loss"]) == T - 1
    assert len(output_sequence.action["params"]["linear"]["w"]) == T - 1


class NonStationaryEnvironment(hk.Module):
    def __call__(self, action, raw_obs):
        step = hk.get_state("step", [], init=lambda *_: 0)

        # Only the environment now the true value of the parameters
        # at step 2000 we flip the sign of the true parameters !
        w_true = hk.cond(
            step < 2000,
            step,
            lambda step: -jnp.ones(3),
            step,
            lambda step: jnp.ones(3),
        )

        # The environment has its proper loss definition
        @jax.jit
        def loss(y_pred, y):
            return jnp.mean(jnp.square(y_pred - y))

        # raw observation contains features and generative noise
        x, noise = raw_obs

        # generate targets
        y = x @ w_true + noise
        obs = (x, y)

        # evaluate the prediction made by the agent
        y_previous = Lag(1)(y)
        y_pred = action["y_pred"]
        reward = loss(y_pred, y_previous)

        step += 1
        hk.set_state("step", step)

        return reward, obs


def test_non_stationary_environement():
    # ## Non-stationary environment
    # Now, let's implement a non-stationary environment
    # Now let's run a gym simulation to see how the agent adapt to the change of environment.

    @hk.transform_with_state
    def gym_fun(raw_obs):
        return GymFeedback(
            linear_regression_agent, NonStationaryEnvironment(), return_action=True
        )(raw_obs)

    T = 300
    raw_observations = generate_many_raw_observations(T)
    rng = jax.random.PRNGKey(42)
    output_sequence, final_state = dynamic_unroll(
        gym_fun,
        None,
        None,
        rng,
        True,
        raw_observations,
    )

    assert len(output_sequence.reward) == T - 1
    assert len(output_sequence.action["loss"]) == T - 1
    assert len(output_sequence.action["params"]["linear"]["w"]) == T - 1
