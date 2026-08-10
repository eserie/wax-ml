# Change log

Best viewed [here](https://eserie.github.io/wax-ml/changelog.html).


<!--
Remember to align the itemized text with the first line of an item within a list.

PLEASE REMEMBER TO CHANGE THE '..main' WITH AN ACTUAL TAG in GITHUB LINK.
-->

## wax 0.7.0 (August 10 2026)

The first release since 0.6.4, and a large one. It adds a Flax backend alongside
the existing Haiku one, moves the package to a `src/` layout, and raises the
supported Python and JAX floors.

### Breaking changes

* **Python 3.11 or later is required** (was 3.9). On an older interpreter `pip`
  refuses the install rather than resolving an old release.
* Core dependency floors move to JAX 0.9, Flax 0.12.4, Haiku 0.0.16, NumPy 2.2 and
  pandas 2.3.
* **pandas is capped below 3.0.** pandas 3 changed the default datetime resolution
  from nanoseconds to microseconds; `wax.stream` whitelists `datetime64[ns]` and
  `wax.encode` casts to it explicitly, so a pandas 3 dataset takes a different
  synchronization path. Supporting both resolutions is planned for a follow-up —
  the cap is there so a fresh install gets a combination that is actually tested,
  rather than discovering the difference at runtime.
* The package moved to a `src/` layout. This changes nothing for users installing
  from PyPI; it matters if you were importing from a checkout.
* `pip install "wax-ml[dev,complete]"` no longer resolves silently to nothing —
  those extras did not exist. See the new extras below.

### New: a Flax backend for streaming

`wax.flax` adds a Flax implementation of the streaming module library (EWMA,
Buffer, ARMA and friends) and one state-threading transform.

`wax.flax.core` exports exactly six names: `FlaxTransformed`,
`flax_transform_with_state`, `flax_unroll`, `flax_unroll_transform`,
`StreamingTransform` and `streaming_transform_with_state`. The Haiku API is
unchanged and remains the supported path for everything else.

`wax.flax.debug` (profiler, memory tracker) and `wax.flax.visualization` (graph
rendering, Jupyter and Bokeh backends, dashboard) ship as optional surfaces behind
their own extras.

**Known limitations, deliberately not exported in 0.7.0.** Three symbols exist in
the tree but are not part of the public surface, because their behaviour does not
match what a caller would reasonably expect:

* `streaming_optimizer` / `StreamingOptimizer` fits a single scalar gain on top of
  a frozen model. It does not update the model's parameters, and the value it
  returns is not the scaled one. Verified by execution, not by reading.
* `streaming_scan`'s reset restores a state that has already consumed the first
  element of the sequence, so history leaks across a reset. A failing test
  (`xfail`) documents this precisely.
* `update_on_event` collides by name with the Haiku `UpdateOnEvent` module while
  differing from it in semantics — its Flax form does not roll back child state.

They remain importable by full module path for anyone who wants to experiment.
Promoting them is 0.8 work.

### Install extras

| Extra | What it enables |
| --- | --- |
| `numba` | `wax.numba` |
| `debug` | `wax.flax.debug` |
| `visualization` | `wax.flax.visualization` |
| `complete` | all of the above, plus the plotting and I/O helpers the notebooks use |
| `optional` | kept unchanged for 0.6.x compatibility |

Each extra now names every *direct* requirement of the submodules it covers;
nothing relies on a dependency arriving transitively.

### Fixes

* `wax.numba`: `DataFrame.wax_numba.ewm(...)` raised `TypeError: got multiple
  values for argument 'accessor'` for any positional argument.
* `wax.flax.debug`: `jax.tree_map` was removed in JAX 0.9; the profiler and memory
  tracker called it.
* `wax.flax.visualization`: `circle(size=...)` was deprecated in Bokeh 3.4, and
  graph rendering dispatched on the available backends before the requested output
  format, so `output_format="text"` reached matplotlib.
* The Colab notebooks' install cell was commented out, so a fresh runtime never
  installed WAX-ML.

### Packaging and project

* Distributions no longer ship the test suite or the demo modules — 58 test files
  and roughly 400 kB left the wheel. `py.typed` is shipped, so annotations are
  visible to type checkers.
* The release workflow now asserts that the tag matches `src/wax/VERSION` and that
  the released commit is reachable from the default branch, then clean-installs and
  smoke-tests the built wheel before publishing it.
* Documentation is built and published from CI to GitHub Pages.
* ruff is the single linter and formatter; flake8, black and isort are gone.

## wax 0.6.4 (April 22 2023)

* upgrade CI to python 3.9
* remove numba functions from the core package
## wax 0.6.3 (November 18 2022)

* import tree_map from tree_util
* fix VMap module
## wax 0.6.2 (November 14 2022)

* Fix deployement of 0.6.1
## wax 0.6.1 (November 14 2022)

* patch vmap_lift_with_state (now rely on hk.experimental.lift_with_state)
## wax 0.6.0 (November 13 2022)

* add stateful.py module with vmap_lift_with_state and unroll_lift_with… (#65)
* remove tensorflow and pytorch from requirements
* remove dependency with eagerpy (#62)
* correct imports of tree utils to avoid FutureWarnings (#63)
* replace jax.tree_multi_map (removed in JAX release v0.3.16) with jax.tree_map
* faster unroll: do not propagate params in scan state (#61)

## wax 0.5.0 (May 16 2022)

* update to Jax v0.3.10 (#59)
* add optimizers  (#56), (#58)

## wax 0.4.0 (April 5 2022)

* EWMA alignement with pandas and speedup (#53)
  This adds the options:
      * `com`
      * `min_periods`
      * `ignore_na`
      * `return_info`
* [wax_numba] add an implementation of the ewma in numba extending the one of pandas with the additional modes we have in wax:
    * `adjust='linear'`
    * `initial_value` parameter
    * a state management for online usages and warm-start of the ewma.
    * add `numba` to requirements

* [EWMA] use `log1com` as a haiku parameter to ease training with gradient descent.
* Align EWMCov and EWMVar with EWMA (#55)

* [PctChange] correct `PctChange` module to align with pandas behavior. Introduce `fillna_zero` option.


## wax 0.3.2 (February 25 2022)

* [modules] faster EWMA in adjust=True mode.

## wax 0.3.1 (January 4 2022)

* [unroll] split rng in two rng keys.

## wax 0.3.0 (December 16 2021)

* [VMap] VMap module works in contexts without PRNG key
* [online optimizer] ; refactor
  * refactor OnlineOptimizer outputs: only return loss, model_info, opt_loss by default.
    New option 'return_params' to return params in outputs
  * OnlineOptimizer returns updated params if return_params is set to True
* [newton optimizer]: use NamedTuple instead of base.OptState
* [unroll] propagate pbar argument to static_scan
* [unroll] Renew the PRNG key in the unroll operations

* refactor usage of OnlineOptimizer in notebooks

* format with laster version of black
* require jax<=0.2.21
* add graphviz to optional dependencies
* upgrade jupytext to 1.13.3
* use python 3.8 in CI and documentation


## wax 0.2.0 (October 20 2021)

* Documentation:
  * New notebook : 07_Online_Time_Series_Prediction
  * New notebook : 08_Online_learning_in_non_stationary_environments

* API modifications:
    * refactor accessors and stream
    * GymFeedback now assumes that agent and env return info object
    * OnlineSupervisedLearner action is y_pred, loss and params are returned as info

* Improvements:
    * introduce general unroll transformation.
    * dynamic_unroll can handle Callable objects
    * UpdateOnEvent can handle any signature for functions
    * EWMCov can handle the x and y arguments explicitly
    * add initial action option to GymFeedback

* New Features:
    * New module UpdateParams
    * New module SNARIMAX, ARMA
    * New module OnlineOptimizer
    * New module VMap
    * add grads_fill_nan_inf option to OnlineSupervisedLearner
    * Introduce `unroll_transform_with_state` following Haiku API.
    * New function auto_format_with_shape and tree_auto_format_with_shape
    * New module Ffill
    * New module Counter

* Deprecate:
    * deprecate dynamic_unroll and static_unroll, refactor their usages.

* Fixes:
    * Simplify Buffer to work only on ndarrays (implementation on pytrees were too complex)
    * EWMA behave corectly with gradient
    * MaskStd behave correctly with gradient
    * correct encode_int64 when working on int32
    * update notebook 06_Online_Linear_Regression and add it to run-notebooks rule
    * correct pct_change to behave correctly when input data has nan values.
    * correct eagerpy test for update of tensorflow, pytorch and jax
    * remove duplicate license comments
    * use numpy.allclose instsead of jax.numpy.allclose for comparaison of non Jax objects
    * update comment in notebooks : jaxlib==0.1.67+cuda111 to jaxlib==0.1.70+cuda111
    * fix jupytext dependency
    * add seaborn as optional dependency


## wax 0.1.0 (June 14 2021)

* First realease.
