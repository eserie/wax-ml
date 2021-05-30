<div align="center">
<img src="https://github.com/eserie/wax-ml/blob/main/docs/_static/wax_logo.png" alt="logo" width="40%"></img>
</div>

# WAX-ML: Machine learning for streaming data
![Continuous integration](https://github.com/eserie/wax-ml/actions/workflows/main.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/wax-ml)
[![Documentation Status](https://readthedocs.org/projects/wax-ml/badge/?version=latest)](https://wax-ml.readthedocs.io/en/latest/)

[**Quickstart**](#quickstart-colab-in-the-cloud)
| [**Install guide**](#installation)
| [**Change logs**](https://wax-ml.readthedocs.io/en/latest/changelog.html)
| [**Reference docs**](https://wax-ml.readthedocs.io/en/latest/)

🌊 Wax is what you put on a surfboard to avoid slipping.  It is an essential tool to go
surfing.  Similarly, WAX-ML strives to be an essential tool for doing machine learning on
streaming data!  🌊

WAX-ML is a research oriented [python](https://www.python.org/) library which provides
tools to design powerful machine learning algorithms working on streaming data.

It strives to complement [JAX](https://jax.readthedocs.io/en/latest/) with tools
dedicated to time-series.

WAX-ML aims to make JAX-based programs easier to use for end users working with [Pandas](https://pandas.pydata.org/) and [Xarray](http://xarray.pydata.org/en/stable/).

## Goal

WAX-ML's goal is to expose "traditional" algorithms that are often difficult to find in
standard python ecosystem and are related to time-series and more generally to streaming
data.

WAX-ML wants to make easy to work with algorithms from very various computational domains
such as machine learning, online learning, reinforcement learning, optimal control,
time-series analysis, optimization, statistical modeling.

For now, WAX-ML focuses on **time series** algorithms as this is one of the areas of
machine learning that lacks the most dedicated tools.  Working with time series is
notoriously known to be difficult and often requires very specific algorithms
(statistical modeling, filtering, optimal control).

Even though some of the modern machine learning tools such as RNN, LSTM or reinforcement
learning can do an excellent job on some specific time series problems, most of the
problems require to keep using more traditional algorithms such as linear and non-linear
filters, FFT, the eigen-decomposition of matrices, Riccati solvers for optimal control
and filtering...

By adopting a "no-framework" approach WAX-ML aim to be an efficient tool to combine modern
machine learning approaches with more traditional ones.

Some work has been done in this direction, for example see [2] where transformer encoder architectures are massively accelerated, with limited accuracy costs, by replacing the self-attention sublayers with a standard, non-parameterized Fast Fourier Transform (FFT).
Their implementation, not yet published, is based on Flax, a tool from the JAX ecosystem.


WAX-ML may also be useful for developing research ideas in areas such as online machine
learning (see [1]) and development of control, reinforcement learning and online-optimization methods.

## What WAX-ML does?

Well, WAX-ML has some pretty ambitious design and implementation goals.

To do things right, we decided to start it small and in an open-source design from the
beginning.

For now, WAX-ML contains:

- transformation tools that we call "unroll" transformations allowing to apply any
  transformation, possibly statefull, on sequential data.  It generalizes the RNN
  architecture to any statefull transformation allowing to implement any kind of
  "filter".

- A "stream" module permitting to synchronize data streams with different time
  resolutions (see [🌊 Streaming Data 🌊](#-streaming-data-))

- pandas and xarray "accessors" permitting to apply any JAX/Haiku module on `DataFrame`,
  `Series`, `Dataset`, `DataArray` data containers.

- Ready-to-use exponential moving average, variance and covariance filters.
  - for JAX users: as Haiku modules (`EWMA`, ... see the complete list in our
  [API documentation](https://wax-ml.readthedocs.io/en/latest/wax.modules.html)
  ),
  - for pandas/xarray users: with drop-in replacement of pandas `ewm` accessor.

- Building blocks for designing of feedback loops in reinforcement learning.  We provide
  a Haiku module called `GymFeedback` that allows the implementation of a Gym feedback
  loops (see [Gym documentation](https://gym.openai.com/)).

- universal functions: we use [EagerPy](https://github.com/jonasrauber/eagerpy) to
  implement "universal" modules that can work with
  [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/),
  [JAX](https://github.com/google/jax) and [NumPy](https://numpy.org/) tensors.  At the
  moment, we only implement a demonstration module: `EagerEWMA`.

## What is JAX?

JAX is a research oriented computational system implemented in python that leverages the
XLA optimization framework for machine learning computations.  It makes XLA usable with
the numpy API and some functional primitives for just-in-time compilation,
differentiation, vectorization and parallelization.  It allows to build higher level
transformations or "programs" in a functional programming approach.


## Why to use WAX-ML?

If you already deal with time series and are a pandas or xarray user, but you want to
use the impressive tools of the JAX ecosystem, then WAX-ML might be the right tool for you,
as it implements pandas and xarray accessors to apply JAX functions.

If you are already a user of JAX, you may be interested in adding WAX-ML to your
toolbox to address time series problems.

## Design

WAX-ML is research oriented library.  It relies on
[JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) and
[Haiku](https://github.com/deepmind/dm-haiku) functional programing paradigm to ease the
development of research ideas.

WAX-ML is a bit like [Flux](https://fluxml.ai/Flux.jl/stable/)
in [Julia](https://julialang.org/) programming language.

WAX-ML is not a framework but either a set of tools which aim to complement [JAX
Ecosystem](https://moocaholic.medium.com/jax-a13e83f49897).

# Contents
* [🚀 Quickstart: Colab in the Cloud 🚀](#-quicksart-colab-in-the-cloud-)
* [🌊 Streaming Data 🌊](#-streaming-data-)
* [🔥 Speed 🔥](#-speed-)
* [⚒ Implementation ⚒](#-Implementation-)
* [Future plans](#future-plans)
* [Disclaimer](#disclaimer)
* [Installation](#installation)
* [Development](#development)
* [References](#references)
* [License](#license)
* [Citing WAX-ML](#citing-wax)
* [Reference documentation](#reference-documentation)


## 🚀 Quickstart 🚀

Jump right in using a notebook in your browser, connected to a Google Cloud GPU or
simply read our notebook in the
[documentation](https://wax-ml.readthedocs.io/en/latest/).

Here are some starter notebooks:
- 〰 Compute exponential moving averages with xarray and pandas accessors 〰 : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/01_demo_EWMA.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/01_demo_EWMA.html)
- ⏱ Synchronize data streams ⏱ : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/02_Synchronize_data_streams.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/02_Synchronize_data_streams.html)
- 🌡 Binning temperatures 🌡 : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/03_ohlc_temperature.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/03_ohlc_temperature.html)
- 🎛 The three steps workflow 🎛 : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/04_The_three_steps_workflow.html)
- 🔭 Reconstructing the light curve of stars with LSTM 🔭: [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/05_reconstructing_the_light_curve_of_stars.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/05_reconstructing_the_light_curve_of_stars.html)


## 🌊 Streaming Data 🌊

WAX-ML may complement JAX ecosystem by adding support for **streaming data**.

To do this, WAX-ML implements a unique **data tracing** mechanism that prepares for fast
access to in-memory data and allows the execution of JAX tractable functions such as
`jit`, `grad`, `vmap`, or `pmap`.

This mechanism is somewhat special in that it works with time series data.

The `wax.stream.Stream` object implements this idea.  It uses python generators to
**synchronize multiple streaming data streams** with potentially different temporal
resolutions.

The `wax.stream.Stream` object works on in-memory data stored in
[`xarray.Dataset`](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html).

To work with "real" streaming data, it should be possible to implement a buffer
mechanism running on any python generator and to use the synchronization and data
tracing mechanisms implemented in WAX-ML to apply JAX transformations on batches of data
stored in memory. (See our WEP4 enhancement proposal)

## ⌛ Adding support for time dtypes in JAX ⌛

At the moment `datetime64`and `string_` dtypes are not supported in JAX.

WAX-ML add support for `datetime64` and `string_` numpy dtypes in JAX.
To do so, WAX-ML implements:
- an encoding scheme for `datetime64` relying on pairs of 32-bit integers similar to `PRNGKey` in JAX.
- an encoding scheme for `string_` relying on `LabelEncoder` of [scikit-learn](https://scikit-learn.org/stable/).

By providing these two encoding schemes, WAX-ML makes it easy to use JAX algorithms on data of these types.

Currently, the types of time offsets supported by WAX-ML are quite limited and we
collaborate with the pandas, xarray and [Astropy](https://www.astropy.org/) teams
to further develop the time manipulation tools in WAX-ML (see "WEP1" in `WEP.md`).

## pandas and xarray accessors

WAX-ML implements pandas and xarray accessors to ease the usage of machine-learning algorithms implemented
with Haiku functions** on
high-level data APIs :
- pandas's `DataFrame` and `Series`
- xarray's `Dataset` and `DataArray`.

To load the accessors, run:
```python
from wax.accessors import register_wax_accessors
register_wax_accessors()
```

Then run the "one-liner" syntax:
```python
<data-container>.stream(…).apply(…)
```

** We call a "Haiku function" any function implemented with Haiku modules.

## Already implemented modules

We have some Haiku modules ready to be used in `wax.modules` (see our [api
documentation](https://wax-ml.readthedocs.io/en/latest/wax.modules.html)).


For now, WAX-ML offers direct access to some modules through specific accessors for xarray
and pandas.
For instance, we have an implementation of the "exponential moving average" directly
accessible through the accessor `<data-container>.ewm(...).mean()` which provides a
drop-in replacement for the exponential moving average of pandas.


For now, WAX-ML offer direct access to some modules through specific accessors for xarray
and pandas.

For instance, you can see our implementation of the "exponential moving average".  This
is a drop-in replacement for the exponential moving average of pandas.


Let's show how it works on the "air temperature" dataset from `xarray.tutorials`:

```python
import xarray as xr
da = xr.tutorial.open_dataset("air_temperature")
dataframe = da.air.to_series().unstack(["lon", "lat"])
```

Pandas ewma:
```python
air_temp_ewma = dataframe.ewm(alpha=1.0 / 10.0).mean()
```

WAX-ML ewma:
```python
air_temp_ewma = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
```


### Apply a custom function to a Dataset

Now let's illustrate how WAX-ML accessors work on [xarray datasets](http://xarray.pydata.org/en/stable/generated/xarray.Dataset.html).

```{code-cell} ipython3
from wax.modules import EWMA


def my_custom_function(da):
    return {
        "air_10": EWMA(1.0 / 10.0)(da["air"]),
        "air_100": EWMA(1.0 / 100.0)(da["air"]),
    }


da = xr.tutorial.open_dataset("air_temperature")
output, state = da.wax.stream().apply(my_custom_function, format_dims=da.air.dims)

_ = output.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
```

![](docs/_static/my_custom_function_on_dataset.png)

You can see our [Documentation](https://wax-ml.readthedocs.io/en/latest/) for examples with
EWMA or Binning on the air temperature dataset.

## ⏱ Synchronize streams ⏱

Physicists, and not the least 😅, have brought a solution to the synchronization
problem.  See [Poincaré-Einstein synchronization Wikipedia
page](https://en.wikipedia.org/wiki/Einstein_synchronisation) for more details.

In WAX-ML we strive to follow their recommendations and implement a synchronization
mechanism between different data streams. Using the terminology of Henri Poincaré (see
link above) we introduce the notion of "local time" to unravel the main stream in which
the user wants to apply transformations. We call the other streams "secondary streams".
They can work at different frequencies, lower or higher.  The data from these secondary
streams will be represented in the "local time" either with the use of a forward filling
mechanism for lower frequencies or a buffering mechanism for higher frequencies.

We implement a "data tracing" mechanism to optimize access to out-of-sync streams.
This mechanism works on in-memory data.  We perform a first pass on the data,
without actually accessing to it, and determine the indices necessary to
later acces to the data. Doing so we are vigilant to not let any "future"
information pass through and thus guaranty a data processing that respects causality.

The buffering mechanism used in the case of higher frequencies, works with a fixed
buffer size (see the WAX-ML module
[`wax.modules.Buffer`](https://wax-ml.readthedocs.io/en/latest/_autosummary/wax.modules.buffer.html#module-wax.modules.buffer))
which allows us to use JAX / XLA optimizations and have efficient processing.

Let's illustrate with a small example how `wax.stream.Stream` synchronizes data streams.

Let's use the dataset "air temperature" with :
- An air temperature defined with hourly resolution.
- A "fake" ground temperature defined with a daily resolution as the air temperature minus 10 degrees.

```python
import xarray as xr
from wax.accessors import register_wax_accessors
from wax.modules import EWMA

register_wax_accessors()


def my_custom_function(da):
  return {
    "air_10": EWMA(1.0 / 10.0)(da["air"]),
    "air_100": EWMA(1.0 / 100.0)(da["air"]),
    "ground_100": EWMA(1.0 / 100.0)(da["ground"]),
  }


da = xr.tutorial.open_dataset("air_temperature")
da["ground"] = da.air.resample(time="d").last().rename({"time": "day"}) - 10
```

```python
results, state = (
	da.wax
	.stream(local_time="time", pbar=True)
	.apply(my_custom_function, format_dims=da.air.dims)
	)
```

```python
results.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
```

![](docs/_static/synchronize_data_streams.png)

## ⚡ Performance on big dataframes ⚡

Check out our [Documentation](https://wax-ml.readthedocs.io/en/latest/) to
see how you can use our "3-step workflow" to speed things up!


# 🔥 Speed 🔥

With WAX-ML, you can already compute an exponential moving average on a 1 millions rows
dataframe with a 3x to 100x speedup
(depending of the data container you use and speed measurement methodology) compared to
pandas implementation.  (See our notebook in the
[Quick Start Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/04_The_three_steps_workflow.html)
or in
[Colaboratory](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)
).

WAX-ML algorithms are implemented in JAX, so they are fast!

The use of JAX allows for algorithm implementations that can be run in a highly
optimized manner on various processing units such as the CPU, GPU and TPU.

WAX-ML does not want to reinvent the wheel by reimplementing every algorithm.  We want
existing machine learning libraries to work well together while trying to leverage their
strength.

# ⚒ Implementation ⚒

Currently, WAX-ML uses
[the Haiku module API](https://dm-haiku.readthedocs.io/en/latest/api.html#modules-parameters-and-state)
and
[Haiku transformation functions](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku-transforms)
to facilitate the development of robust and reusable features.

Haiku's module API integrates well with the functional paradigm of JAX and makes it easy
to develop "mini-languages" tailored to specific scientific domains.

** [Flax](https://github.com/google/flax)
also has a module API. We should consider using it in WAX-ML too!


## Universal functions

WAX-ML uses [eagerpy](https://github.com/jonasrauber/eagerpy) to efficiently mix different
types of tensors and develop high level APIs to work with
e.g. [numpy](https://numpy.org/), [pandas](https://pandas.pydata.org/),
[xarray](http://xarray.pydata.org/en/stable/).

The use of eagerpy should allow, if necessary, to propose implementations of algorithms
compatible with other tensor libraries such as numpy, tensorflow and pytorch, with
native performance.

We currently have a working example for the EWMA module which is implemented in
`wax.universal.modules`.  See the code in :
```
wax.universal.modules.ewma.py
wax.universal.modules.ewma_test.py
```

For now, the core algorithms in WAX-ML are only implemented in JAX in order to "stay
focused".  But if there is interest in implementing universal algorithms, more work
could be done from this simple example.

Currently, WAX-ML uses an unpublished
[fork of eagerpy](https://github.com/eserie/eagerpy/tree/dev).  We have opened some pull requests
in eagerpy that should allow us to go to the official eagerpy library as soon as they
are accepted.

## Gym Feedback

WAX-ML implements **feedback loops** which are very natural when working with time series.

For now, we only implement Gym feedback between an *agent* and an *environment* exposed in
the JAX / Haiku functional API.

WAX-ML implements a `GymFeedback` module which is built from an agent and an environment:
- An *agent* is a module with an `observation` input and an `action` output.
- An *environment* is a module with a pair `(action, raw observation)` as input
  and a pair `(reward, observation)` as output.

A feedback instance `GymFeedback(agent, env)` is a module with a `raw observation` input
and a `reward` output.

`GymFeedback` is implemented as a regular Haiku module in `wax.modules`.

In addition, to ensure compatibility with other tools in the Gym ecosystem, we propose a
*transformation* mechanism to transform functions into standard stateful python objects
following the Gym API for *agents* and *environments* implemented in
[deluca](https://scikit-learn.org/stable/).  These wrappers are in the `wax.gym`
package.

WAX-ML implements *callbacks* in the `wax.gym` package.  The callback API was inspired by
the one in the one in [dask](https://github.com/dask/dask).

We would like to collaborate with the deluca team to develop this part of WAX-ML.

The JAX ecosystem already has a library dedicated to reinforcement learning:
[RLax](https://github.com/deepmind/rlax).  What is done in WAX-ML could be transferred to
it.

# Future plans

## Feedback loops and Control Toolbox

We would like to implement other types of feedback loops in WAX-ML.
For instance, see:
[python-control](https://github.com/python-control/python-control),
[Slycot](https://github.com/python-control/Slycot).

Many algorithms in this space are absent from the python ecosystem.  If they are not
implemented in other libraries, WAX-ML aims to implement them via JAX and expose them with
a simple API.


An idiomatic example is the [Kalman
filter](https://fr.wikipedia.org/wiki/Filtre_de_Kalman), a now-standard algorithm that
dates back to the 1950s.  After 30 years of existence, the Python ecosystem has still
not integrated this algorithm into a standard library!

Some implementations can be found in
[python-control](https://github.com/python-control/python-control),
[stats-models](https://www.statsmodels.org/stable/index.html), [SciPy
Cookbook](https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html#).  Some
machine learning libraries have closed non-solved issues like [scikit-learn
#862](https://github.com/scikit-learn/scikit-learn/pull/862) or [river
#355](https://github.com/online-ml/river/pull/355).

Why didn't the Kalman filter find its place in these libraries?  Because they have an
object oriented API (which is very good!)  offering them specific APIs very well adapted
to specific problems of modern machine learning.

The functional approach of WAX-ML, inherited from JAX, could well help to integrate a
Kalman filter implementation in a machine learning ecosystem.  (See WEP5 for the
integration plan)

In fact, it turns out that python code written with JAX is not very far from from
[Fortran](https://fr.wikipedia.org/wiki/Fortran), a (mathematical FORmula TRANslating
system).  It should therefore be quite easy and natural to reimplement standard
algorithms implemented in Fortran, such as those in the
[Slycot](https://github.com/python-control/Slycot) and [SLICOT](http://slicot.org/)
libraries.

In fact, as noted in [this issue of
JAX](https://github.com/google/jax/discussions/3950), it might even be possible to
simply wrap Fortran code in JAX.  This would avoid a painful rewriting process!


## Optimization

JAX ecosystem has already a library dedicated to optimization:
[Optax](https://github.com/deepmind/optax).  It may be interested to complement it with
other first order algorithms such as [ADMM](https://stanford.edu/~boyd/admm.html).

One can find "functional" implementations of proximal algorithms in python (see
[proxmin](https://github.com/pmelchior/proxmin)) and Julia (see
[ProximalOperators](https://github.com/kul-forbes/ProximalOperators.jl),
[COSMO](https://github.com/oxfordcontrol/COSMO.jl) ). Collaborating with these teams
should be very interesting.

Other type of works took place around automatic differentiation and
optimization such as those conducted in [cvxpylayers](https://github.com/cvxgrp/cvxpylayers)
which proposes a mechanism to use convex optimizers in
differentiable layers.
They have already implemented a JAX API but they can't use the `jit` compilation of JAX at the moment
(see [this issue](https://github.com/cvxgrp/cvxpylayers/issues/103)). We would be interested to help for
that!


## Other algorithms

The machine learning libraries [scikit-learn](https://scikit-learn.org/stable/),
[river](https://github.com/online-ml/river),
[ml-numpy](https://github.com/ddbourgin/numpy-ml) implement many "traditional" machine
learning algorithms that should provide an excellent basis for linking or reimplementing
in JAX.

WAX-ML could be the place where we reimplement some of these algorithms to work with the
JAX ecosystem.


## Other APIS

As it did for the Gym API, WAX-ML could add support for other high-level OO APIs like
Keras, scikit-learn, river ...


## Collaborations

The WAX-ML team is open to discussion and collaboration with contributors from any field
who interested in using WAX-ML for their problems on streaming data.  We are looking for
use cases around data streaming in audio processing, natural language processing,
astrophysics, biology, engineering ...

We believe that good software design, especially in the scientific domain, requires
practical use cases and that the more diversified these use cases are, the more the
developed functionalities will be guaranteed to be well implemented.

We expect interactions with the projects cited in this README.  Do not hesitate to
contact us if you think your project can have fruitful interactions with WAX-ML.

By making this software public, we hope to find enthusiasts who aim to develop WAX-ML
further!

# Installation

For now, WAX-ML can only be installed from sources:

```bash
pip install "wax-ml[dev,complete] @ git+https://github.com/eserie/wax-ml.git"
```

# Disclaimer

WAX-ML is in its early stages of development and its features and API are very likely to
evolve.


# Development

You can contribute to WAX-ML by asking questions, proposing practical use cases or by
contributing to the code or the documentation.  You can have a look at our [Contributing
Guidelines](https://github.com/eserie/wax-ml/CONTRIBUTING.md) and [Developer
Documentation](https://wax-ml.readthedocs.io/en/latest/developer.html) .

We maintain a "WAX-ML Enhancement Proposals" in
[WEP.md](https://github.com/eserie/wax-ml/WEP.md) file.


# References

[1] [Google Princeton AI and Hazan Lab @ Princeton University](https://www.minregret.com/research/)

[2] ["FNet: Mixing Tokens with Fourier Transforms", James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon](https://arxiv.org/abs/2105.03824)


# License

```
Copyright 2021 The WAX-ML Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

WAX-ML bundles portions of astropy, dask, deluca, jax, xarray.

dask, astropy are available under a "3-clause BSD" license:
- dask: `wax/gym/callbacks/callbacks.py`
- astropy: `CONTRIBUTING.md`

deluca, jax and xarray are  available under a "Apache" license:
- deluca: `wax/gym/entity.py`
- jax: `docs/conf.py`, `docs/developer.md`
- xarray: `wax/datasets/generate_temperature_data.py`

The full text of these `licenses` are included in the licenses directory.


## Citing WAX-ML

To cite this repository:

```
@software{wax-ml2021github,
  author = {Emmanuel Sérié},
  title = {{WAX-ML}: A {P}ython library for machine-learning on streaming data},
  url = {http://github.com/eserie/wax-ml},
  version = {0.0.2},
  year = {2021},
}
```

## Reference documentation

For details about the WAX-ML API, see the
[reference documentation](https://wax-ml.readthedocs.io/en/latest/).
