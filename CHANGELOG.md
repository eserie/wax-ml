# Change log

Best viewed [here](https://wax-ml.readthedocs.io/en/latest/changelog.html).


<!--
Remember to align the itemized text with the first line of an item within a list.

PLEASE REMEMBER TO CHANGE THE '..main' WITH AN ACTUAL TAG in GITHUB LINK.
-->

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
