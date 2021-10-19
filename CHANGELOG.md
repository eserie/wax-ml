# Change log

Best viewed [here](https://wax-ml.readthedocs.io/en/latest/changelog.html).


<!--
Remember to align the itemized text with the first line of an item within a list.

PLEASE REMEMBER TO CHANGE THE '..main' WITH AN ACTUAL TAG in GITHUB LINK.
-->

**## wax 0.2.0 (October 19 2021)

API modifications:
    - refactor accessors and stream
    - GymFeedback now assumes that agent and env return info object
    - OnlineSupervisedLearner action is y_pred, loss and params are returned as info

Improvements:
    - introduce general unroll transformation.
    - dynamic_unroll can handle Callable objects
    - UpdateOnEvent can handle any signature for functions
    - EWMCov can handle the x and y arguments explicitly
    - add initial action option to GymFeedback

New Features:
    - New module UpdateParams
    - New module SNARIMAX, ARMA
    - New module OnlineOptimizer
    - New module VMap
    - add grads_fill_nan_inf option to OnlineSupervisedLearner
    - Introduce `unroll_transform_with_state` following Haiku API.
    - New function auto_format_with_shape and tree_auto_format_with_shape
    - New module Ffill
    - New module Counter

Deprecate:
    - deprecate dynamic_unroll and static_unroll, refactor their usages.

Fixes:
    - Simplify Buffer to work only on ndarrays (implementation on pytrees were too complex)
    - EWMA behave corectly with gradient
    - MaskStd behave correctly with gradient
    - correct encode_int64 when working on int32
    - update notebook 06_Online_Linear_Regression and add it to run-notebooks rule
    - correct pct_change to behave correctly when input data has nan values.
    - correct eagerpy test for update of tensorflow, pytorch and jax
    - remove duplicate license comments
    - use numpy.allclose instsead of jax.numpy.allclose for comparaison of non Jax objects
    - update comment in notebooks : jaxlib==0.1.67+cuda111 to jaxlib==0.1.70+cuda111
    - fix jupytext dependency
    - add seaborn as optional dependency


## wax 0.1.0 (June 14 2021)

* First realease.
