<div align="center">
<img src="https://github.com/eserie/wax-ml/blob/main/docs/_static/wax_logo.png" alt="logo" width="40%"></img>
</div>

# WAX: Machine learning for streaming data
![Continuous integration](https://github.com/eserie/wax-ml/actions/workflows/main.yml/badge.svg)
![PyPI version](https://img.shields.io/pypi/v/wax-ml)
[![Documentation Status](https://readthedocs.org/projects/wax-ml/badge/?version=latest)](https://wax-ml.readthedocs.io/en/latest/)

[**Quickstart**](#quickstart-colab-in-the-cloud)
| [**Install guide**](#installation)
| [**Change logs**](https://wax-ml.readthedocs.io/en/latest/changelog.html)
| [**Reference docs**](https://wax-ml.readthedocs.io/en/latest/)

🌊 Wax is what you put on a surfboard to avoid slipping.  It is an essential tool to go
surfing.  Similarly, WAX strives to be an essential tool for doing machine learning on
streaming data!  🌊

WAX is a research oriented [python](https://www.python.org/) library which provides
tools to design powerful machine learning algorithms working on streaming data.

It strives to complement [JAX](https://jax.readthedocs.io/en/latest/) with tools
dedicated to time-series.

WAX aims to make JAX-based programs easier to use for end users working with [Pandas](https://pandas.pydata.org/) and [Xarray](http://xarray.pydata.org/en/stable/).

## Goal

WAX's goal is to expose "traditional" algorithms that are often difficult to find in
standard python ecosystem and are related to time-series and more generally to streaming
data.

WAX wants to make easy to work with algorithms from very various computational domains
such as machine learning, online learning, reinforcement learning, optimal control,
time-series analysis, optimization, statistical modeling.

For now, WAX focuses on **time series** algorithms as this is one of the areas of
machine learning that lacks the most dedicated tools.  Working with time series is
notoriously known to be difficult and often requires very specific algorithms
(statistical modeling, filtering, optimal control).

Even though some of the modern machine learning tools such as RNN, LSTM or reinforcement
learning can do an excellent job on some specific time series problems, most of the
problems require to keep using more traditional algorithms such as linear and non-linear
filters, FFT, the eigen-decomposition of matrices, Riccati solvers for optimal control
and filtering...

By adopting a "no-framework" approach WAX aim to be an efficient tool to combine modern
machine learning approaches with more traditional ones.

Some recent work have been made in this directions, for instance see [2].

WAX may also be useful for developing research ideas in areas such as *online machine
learning* (see [1]).

## What is WAX?

Well, WAX has some pretty ambitious design and implementation goals.

To do things right, we decided to start small and in an open-source design from the
beginning.

For now, WAX contains:

- transformation tools that we call "unroll" transformations allowing to apply any
  transformation, possibly statefull, on sequential data.  It generalizes the RNN
  architecture to any statefull transformation allowing to implement any kind of
  "filter".

- A "stream" module permitting to synchronize data streams with different time
  resolutions (see [🌊 Streaming Data 🌊](#-streaming-data-))

- Pandas and Xarray "accessors" permitting to apply any JAX/Haiku module on `DataFrame`,
  `Series`, `Dataset`, `DataArray` data containers.

- Ready-to-use exponential moving average, variance and covariance filters.
  - for JAX users: as Haiku modules (`EWMA`, ... see the complete list in our
  [API documentation](https://wax-ml.readthedocs.io/en/latest/wax.modules.html)
  ),
  - for pandas/xarray users: with drop-in replacement of pandas `ewm` accessor.

- Building blocks for designing of feedback loops in reinforcement learning.  We provide
  a Haiku module called `GymFeedback` that allows the implementation of a "gym feedback
  loops" (see [Gym documentation](https://gym.openai.com/)).

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


## Why to use WAX?

If you already deal with time series and are a pandas or xarray user, but you want to
use the impressive tools of the JAX ecosystem, then WAX might be the right tool for you,
as it implements xarray and pandas accessors to apply JAX functions.

If you are already a fan and/or user of JAX, you may be interested in adding WAX to your
toolbox to address time series problems.

## Design

WAX is research oriented library.  It relies on
[JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) and
[Haiku](https://github.com/deepmind/dm-haiku) functional programing paradigm to ease the
development of research ideas.

WAX is a bit like [Flux](https://fluxml.ai/Flux.jl/stable/)
in [Julia](https://julialang.org/) programming language.

WAX is not a framework but either a set of tools which aim to complement [JAX
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
* [Citing WAX](#citing-wax)
* [Reference documentation](#reference-documentation)


## 🚀 Quickstart 🚀

Jump right in using a notebook in your browser, connected to a Google Cloud GPU or
simply read our notebook in the
[documentation](https://wax-ml.readthedocs.io/en/latest/).

Here are some starter notebooks:
- 01_demo_EWMA : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/01_demo_EWMA.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/01_demo_EWMA.html)
- 02_Synchronize_data_streams : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/02_Synchronize_data_streams.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/02_Synchronize_data_streams.html)
- 03_ohlc_temperature : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/03_ohlc_temperature.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/03_ohlc_temperature.html)
- 04_The_three_steps_workflow : [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb),
  [Open in Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/04_The_three_steps_workflow.html)


## 🌊 Streaming Data 🌊

WAX may complement JAX ecosystem by adding support for **streaming data**.

To do this, WAX implements a unique **data tracing** mechanism that prepares for fast
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
tracing mechanisms implemented in WAX to apply JAX transformations on batches of data
stored in memory. (See our WEP4 enhancement proposal)

## ⌛ Adding support for time dtypes in JAX ⌛

At the moment `datetime64`and `string_` dtypes are not supported in JAX.

WAX add support for `datetime64` and `string_` numpy dtypes in JAX.
To do so, WAX implements:
- an encoding scheme for `datetime64` relying on pairs of 32-bit integers similar to `PRNGKey` in JAX.
- an encoding scheme for `string_` relying on `LabelEncoder` of [scikit-learn](https://scikit-learn.org/stable/).

Since JAX does not currently support the `datetime64` and `string_` dtypes, this WAX
encoding scheme allows easily use JAX algorithms on data with these dtypes.

Currently, the types of temporal representations supported by WAX are quite limited, we
should collaborate with the pandas, xarray and [Astropy](https://www.astropy.org/) teams
to further develop time manipulation tools in WAX. (see "WEP1" in `WEP.md`).

## Accessors

WAX comes with accessors to run Haiku functions** on xarray (`Dataset`, `DataArray`) and
pandas (`DataFrame`, `Series`) data containers.

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

We have some Haiku modules ready to be used in `wax.modules` (see our [api
documentation](https://wax-ml.readthedocs.io/en/latest/wax.modules.html)).

For now, WAX offer direct access to some modules through specific accessors for xarray
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

WAX ewma:
```python
air_temp_ewma = dataframe.wax.ewm(alpha=1.0 / 10.0).mean()
```

WAX accessors also permits to work as easily on xarray datasets.
You can see our [Documentation](https://wax-ml.readthedocs.io/en/latest/) for examples with
EWMA or Binning on the air temperature dataset.

## ⏱ Synchronize streams ⏱

WAX makes it easy to synchronize data streams with different time resolutions.

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
	.stream(master_time="time", pbar=True)
	.apply(my_custom_function, format_dims=da.air.dims)
	)
```

```python
results.isel(lat=0, lon=0).drop(["lat", "lon"]).to_pandas().plot(figsize=(12, 8))
```

## ⚡ Performance on big dataframes ⚡

Check out our [Documentation](https://wax-ml.readthedocs.io/en/latest/) to
see how you can use our "3-step workflow" to speed things up!


# 🔥 Speed 🔥

With WAX, you can already compute an exponential moving average on a 1 millions rows
dataframe with a 2x to 130x speedup (depending of the datacontainer you use) compared to
pandas implementation.  (See our notebook in the
[Quick Start Documentation](https://wax-ml.readthedocs.io/en/latest/notebooks/04_The_three_steps_workflow.html)
or in
[Colaboratory](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/04_The_three_steps_workflow.ipynb)
).

WAX algorithms are implemented in JAX, so they are fast!

The use of JAX allows for algorithm implementations that can be run in a highly
optimized manner on various processing units such as the CPU, GPU and TPU.

WAX does not want to reinvent the wheel by reimplementing every algorithm.  We want
existing machine learning libraries to work well together while trying to leverage their
strength.

# ⚒ Implementation ⚒

Currently, WAX uses
[the Haiku module API](https://dm-haiku.readthedocs.io/en/latest/api.html#modules-parameters-and-state)
and
[Haiku transformation functions](https://dm-haiku.readthedocs.io/en/latest/api.html#haiku-transforms)
to facilitate the development of robust and reusable features.

Haiku's module API integrates well with the functional paradigm of JAX and makes it easy
to develop "mini-languages" tailored to specific scientific domains.

** [Flax](https://github.com/google/flax)
also has a module API. We should consider using it in WAX too!


## Universal functions

WAX uses [eagerpy](https://github.com/jonasrauber/eagerpy) to efficiently mix different
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

For now, the core algorithms in WAX are only implemented in JAX in order to "stay
focused".  But if there is interest in implementing universal algorithms, more work
could be done from this simple example.

Currently, WAX uses an unpublished
[fork of eagerpy](https://github.com/eserie/eagerpy/tree/dev).  We have opened some pull requests
in eagerpy that should allow us to go to the official eagerpy library as soon as they
are accepted.

## Gym Feedback

Wax implements **feedback loops** which are very natural when working with time series.

For now, we only implement Gym feedback between an *agent* and an *environment* exposed in
the JAX / Haiku functional API.

WAX implements a `GymFeedback` module which is built from an agent and an environment:
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

WAX implements *callbacks* in the `wax.gym` package.  The callback API was inspired by
the one in the one in [dask](https://github.com/dask/dask).

We would like to collaborate with the deluca team to develop this part of WAX.

The JAX ecosystem already has a library dedicated to reinforcement learning:
[RLax](https://github.com/deepmind/rlax).  What is done in WAX could be transferred to
it.

# Future plans

## 🔭 Reconstructing the light curve of stars 🔭

To illustrate how more advanced machine learning algorithms can be used on time series,
we would like to reproduce
[this blog post](https://towardsdatascience.com/how-to-use-deep-learning-for-time-series-forecasting-3f8a399cf205)
on star light curve reconstruction techniques.

## Feedback loops and Control Toolbox

We would like to implement other types of feedback loops in WAX.
For instance, see:
[python-control](https://github.com/python-control/python-control),
[Slycot](https://github.com/python-control/Slycot).

Many algorithms in this space are absent from the python ecosystem.  If they are not
implemented in other libraries, WAX aims to implement them via JAX and expose them with
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

The functional approach of WAX, inherited from JAX, could well help to integrate a
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

WAX could be the place where we reimplement some of these algorithms to work with the
JAX ecosystem.


## Other APIS

As it did for the Gym API, WAX could add support for other high-level OO APIs like
Keras, scikit-learn, river ...


## Collaborations

The WAX team is open to discussion and collaboration with contributors from any field
who interested in using WAX for their problems on streaming data.  We are looking for
use cases around data streaming in audio processing, natural language processing,
astrophysics, biology, engineering ...

We believe that good software design, especially in the scientific domain, requires
practical use cases and that the more diversified these use cases are, the more the
developed functionalities will be guaranteed to be well implemented.

We expect interactions with the projects cited in this README.  Do not hesitate to
contact us if you think your project can have fruitful interactions with WAX.

By making this software public, we hope to find enthusiasts who aim to develop WAX
further!

# Installation

For now, WAX can only be installed from sources:

```bash
pip install "wax-ml[dev,complete] @ git+https://github.com/eserie/wax-ml.git"
```

# Disclaimer

WAX is in its early stages of development and its features and API are very likely to
evolve.


# Development

You can contribute to WAX by asking questions, proposing practical use cases or by
contributing to the code or the documentation.  You can have a look at our [Contributing
Guidelines](https://github.com/eserie/wax-ml/CONTRIBUTING.md) and [Developer
Documentation](https://wax-ml.readthedocs.io/en/latest/developer.html) .

We maintain a "WAX Enhancement Proposals" in
[WEP.md](https://github.com/eserie/wax-ml/WEP.md) file.


# References

[1] [Google Princeton AI and Hazan Lab @ Princeton University](https://www.minregret.com/research/)

[2] ["FNet: Mixing Tokens with Fourier Transforms", James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon](https://arxiv.org/abs/2105.03824)


# License

```
Copyright 2021 The Wax Authors

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

WAX bundles portions of astropy, dask, deluca, jax, xarray.

dask, astropy are available under a "3-clause BSD" license:
- dask: `wax/gym/callbacks/callbacks.py`
- astropy: `CONTRIBUTING.md`

deluca, jax and xarray are  available under a "Apache" license:
- deluca: `wax/gym/entity.py`
- jax: `docs/conf.py`, `docs/developer.md`
- xarray: `wax/datasets/generate_temperature_data.py`

The full text of these `licenses` are included in the licenses directory.


## Citing WAX

To cite this repository:

```
@software{wax-ml2021github,
  author = {Emmanuel Sérié},
  title = {{WAX}: A {P}ython library for machine-learning on streaming data based on JAX},
  url = {http://github.com/eserie/wax-ml},
  version = {0.0.2},
  year = {2021},
}
```

## Reference documentation

For details about the WAX API, see the
[reference documentation](https://wax-ml.readthedocs.io/en/latest/).
