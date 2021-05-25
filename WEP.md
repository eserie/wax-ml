
## WEP1

WAX wants to extend Jax with time series features like those developed in pandas,
xarray and astropy.

To ease the execution of workloads in JAX-XLA with a 32-bit float configuration,
we propose an encoding scheme for datetime64 as 32-bit integer pairs
similar to the PRNG keys used by Jax to resume sampling pseudo-random numbers.
In the same way that haiku introduces a generator to ease sampling of PRNG key sequences,
WAX should provide a similar mechanism for generating sequences
of encoded times according to a given scheme (sampling at a certain frequency, ...).

## WEP2

Find a way to perform JAX's "code tracing" only once when using pandas and xarray accessors.

## WEP3

The formatting step from JAX arrays to pandas/xarray data containers is quite slow currently.
We think that working with pandas/xarray teams, we could make this step much faster.

This may be related to this [xarray issue](https://github.com/pydata/xarray/issues/2799)
about performance when doing machine learning on datasets.

## WEP4

To work with "real" streaming data, it should be possible to implement a buffer mechanism running
on any python generator and to use the synchronization and data tracing mechanisms implemented in WAX to apply JAX
transformations on batches of data stored in memory.

Design a `wax.dataset` high level API to stream data.
This should be well integrated with tensorflow `tf.dataset` library.

## WEP5

WAX can implement other types of feedback loops, such as those typically
implemented in standard control "tool boxes."
For instance, see:
[python-control](https://github.com/python-control/python-control),
[Slycot](https://github.com/python-control/Slycot).

Many algorithms in this space are absent from the python ecosystem.
If they are not implemented in other libraries,
WAX aims to implement them via JAX and expose them with a simple API.

An idiomatic example is the [Kalman filter](https://fr.wikipedia.org/wiki/Filtre_de_Kalman),
a now-standard algorithm that dates back to the 1950s.
After 30 years of existence, the Python ecosystem has still not integrated this algorithm into
a standard library!

Some attempts have been made in excellent machine learning libraries. See for instance
[SciPy Cookbook](https://scipy-cookbook.readthedocs.io/items/KalmanFiltering.html#) or
closed issues  in [scikit-learn #862](https://github.com/scikit-learn/scikit-learn/pull/862)
or [river #355](https://github.com/online-ml/river/pull/355).

Why didn't the Kalman filter find its place in these libraries?
Because they have an object oriented API (which is very good!)
offering them specific APIs very well adapted to specific problems
of modern machine learning.

The functional approach of WAX, inherited from JAX,
could well help to solve this 30 years old problem in the python ecosystem :smiley: !



# WEP6
Add TensorBorad callback for Gym API
