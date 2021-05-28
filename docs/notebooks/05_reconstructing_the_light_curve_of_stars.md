---
jupytext:
  encoding: '# -*- coding: utf-8 -*-'
  formats: ipynb,py,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 🔭 Reconstructing the light curve of stars 🔭

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eserie/wax-ml/blob/main/docs/notebooks/05_reconstructing_the_light_curve_of_stars.ipynb)
In Colab install wax by executing this line in a cell:
```python
! pip install "wax-ml[dev,complete] @ git+https://github.com/eserie/wax-ml.git"
```
+++

Let's take a walk through the stars...

This notebook is based on the study done in
[this post by Christophe Pere](https://towardsdatascience.com/how-to-use-deep-learning-for-time-series-forecasting-3f8a399cf205)
and the notebook available on
[the authors's github](https://github.com/Christophe-pere/Time_series_RNN).

We will repeat this study on starlight using the LSTM architecture to predict the observed light flux through time.

Our LSTM implementation is based on this [notebook from Haiku's github repository](https://github.com/deepmind/dm-haiku/blob/master/examples/haiku_lstms.ipynb).

We will see how to use WAX to facilitate the preparation of time series data stored in dataframes and having Nans
before calling a "standard" deep-learning workflow.

+++

## Download the data

```{code-cell} ipython3
%matplotlib inline
```

```{code-cell} ipython3
import io
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
```

```{code-cell} ipython3
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

from wax.accessors import register_wax_accessors
from wax.modules import RollingMean

register_wax_accessors()
```

```{code-cell} ipython3
%%time
# Downloading the csv file from Chrustioge Pere GitHub account
download = requests.get(
    "https://raw.github.com/Christophe-pere/Time_series_RNN/master/kep_lightcurves.csv"
).content
raw_dataframe = pd.read_csv(io.StringIO(download.decode("utf-8")))
raw_dataframe.index = pd.Index(
    pd.date_range("2009-03-07", periods=len(raw_dataframe.index), freq="h"), name="time"
)
```

Let's visualize the description of this dataset:

```{code-cell} ipython3
raw_dataframe.describe().T.to_xarray()
```

```{code-cell} ipython3
stars = raw_dataframe.columns
stars = sorted(list(set([i.split("_")[0] for i in stars])))
print(f"The number of stars available is: {len(stars)}")
print(f"star identifiers: {stars}")
```

```{code-cell} ipython3
dataframe = raw_dataframe[[i + "_rscl" for i in stars]].rename(
    columns=lambda c: c.replace("_rscl", "")
)
dataframe.columns.names = ["star"]
dataframe.shape
```

## Rolling mean

+++

We will smooth the data by applying a rolling mean with a window of 100 periods.

+++

### Count nan values

But before since the dataset has some nan values, we will extract few statistics
about the density of nan values in windows of size 100.

It will be the occasion to show a usage of the `wax.modules.Buffer` module with the `format_outputs=False`
option for the dataframe accessor `.wax.stream`.

```{code-cell} ipython3
import jax.numpy as jnp
import numpy as onp
```

```{code-cell} ipython3
from wax.modules import Buffer
```

Let's apply the `Buffer` module to the data:

```{code-cell} ipython3
buffer, _ = dataframe.wax.stream(format_outputs=False).apply(lambda x: Buffer(100)(x))
```

```{code-cell} ipython3
assert isinstance(buffer, jnp.ndarray)
```

Let's describe the statistic of nans with pandas:

```{code-cell} ipython3
count_nan = jnp.isnan(buffer).sum(axis=1)
pd.DataFrame(onp.array(count_nan)).stack().describe().astype(int)
```

### Computing the rolling mean

+++

We will choose a `min_periods` of 5 in order to keep at leas 75% of the points.

```{code-cell} ipython3
%%time
dataframe_mean, _ = dataframe.wax.stream().apply(
    lambda x: RollingMean(100, min_periods=5)(x)
)
```

```{code-cell} ipython3
dataframe.loc[:, "008241079"].plot()
dataframe_mean.loc[:, "008241079"].plot()
```

### With Dataset API

Let's illustrate how to do the same rolling mean operation but using wax accessors on xarray `Dataset`.

```{code-cell} ipython3
from functools import partial

from jax.tree_util import tree_map
```

```{code-cell} ipython3
dataset = dataframe.to_xarray()
```

```{code-cell} ipython3
dataset
```

```{code-cell} ipython3
%%time
dataset_mean, _ = dataset.wax.stream().apply(
    partial(tree_map, lambda x: RollingMean(100, min_periods=5)(x)),
    format_dims=["time"],
)
```

(Its much longer than with dataframe)

TODO: This is an issue that we should solve.

```{code-cell} ipython3
dataset_mean
```

```{code-cell} ipython3
dataset["008241079"].plot()
dataset_mean["008241079"].plot()
```

### With dataarray

```{code-cell} ipython3
dataarray = dataframe.to_xarray().to_array("star").transpose("time", "star")
```

```{code-cell} ipython3
dataarray
```

```{code-cell} ipython3
%%time
dataarray_mean, _ = dataarray.wax.stream().apply(
    lambda x: RollingMean(100, min_periods=5)(x)
)
```

(Its much longer than with dataframe)

```{code-cell} ipython3
dataarray_mean
```

```{code-cell} ipython3
dataarray.sel(star="008241079").plot()
dataarray_mean.sel(star="008241079").plot()
```

## Forecasting with Machine Learning

We need two forecast in this data, if you look with attention you'll see micro holes and big holes.

```{code-cell} ipython3
import warnings
from typing import NamedTuple, Tuple, TypeVar

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import plotnine as gg

T = TypeVar("T")
Pair = Tuple[T, T]


class Pair(NamedTuple):
    x: T
    y: T


class TrainSplit(NamedTuple):
    train: T
    validation: T


gg.theme_set(gg.theme_bw())
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 18, 8
fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]})
dataframe.plot(ax=ax, title="raw data")
ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=lax.transAxes)
lax.axis("off")
```

```{code-cell} ipython3
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = 18, 8
fig, (ax, lax) = plt.subplots(ncols=2, gridspec_kw={"width_ratios": [4, 1]})
dataframe_mean.plot(ax=ax, title="Smoothed data")
ax.legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=lax.transAxes)
lax.axis("off")
```

```{code-cell} ipython3
star = "007609553"
print(f"Look at star: {star}")
values = dataframe_mean[star].values
```

```{code-cell} ipython3
values.shape
```

```{code-cell} ipython3
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(values[~np.isnan(values)].reshape(-1, 1))
dataset.shape
```

```{code-cell} ipython3
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]
```

reshape into X=t and Y=t+1

```{code-cell} ipython3
def create_dataset(values, look_back=1):
    """
    Function to prepare a list of (x, y) data points to data for time series learning
    @param values: (list) list of values
    @param look_back: (int) number of values for the x list [x1, x2, x3, ... , xn] default 1
    @return _x: x values for the time series
    @return _y: y values for the time series
    """
    # set empty lists
    _x, _y = [], []
    for i in range(len(values) - look_back - 1):
        a = values[i : (i + look_back)]  # stack a list of values
        _x.append(a)  # set x
        _y.append(values[i + look_back])  # set y
    return np.array(_x), np.array(_y)
```

```{code-cell} ipython3
look_back = 20
train_x, train_y = create_dataset(train, look_back)
test_x, test_y = create_dataset(test, look_back)
```

```{code-cell} ipython3
train_x.shape, train_y.shape, test_x.shape, test_y.shape
```

### Normalize data

```{code-cell} ipython3
dataframe_mean.stack().hist(bins=100)
```

```{code-cell} ipython3
from wax.encode import Encoder


def min_max_scaler(values: pd.DataFrame, output_format: str = "dataframe") -> Encoder:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(values)
    index = values.index
    columns = values.columns

    def encode(dataframe: pd.DataFrame):
        nonlocal index
        nonlocal columns
        index = dataframe.index
        columns = dataframe.columns
        array_normed = scaler.transform(dataframe)
        if output_format == "dataframe":
            return pd.DataFrame(array_normed, index, columns)
        elif output_format == "jax":
            return jnp.array(array_normed)
        else:
            return array_normed

    def decode(array_scaled):
        value = scaler.inverse_transform(array_scaled)
        if output_format == "dataframe":
            return pd.DataFrame(value, index, columns)
        else:
            return value

    return Encoder(encode, decode)
```

```{code-cell} ipython3
scaler = min_max_scaler(dataframe_mean)
dataframe_normed = scaler.encode(dataframe_mean)
assert (scaler.decode(dataframe_normed) - dataframe_mean).stack().abs().max() < 1.0e-4
```

```{code-cell} ipython3
dataframe_normed.stack().hist(bins=100)
```

### Prepare train / validation datasets

```{code-cell} ipython3
from wax.modules import FillNanInf, Lag
```

```{code-cell} ipython3
SEQ_LEN = 64
```

```{code-cell} ipython3
def split_feature_target(dataframe, look_back=SEQ_LEN) -> Pair:
    x, _ = dataframe.wax.stream(format_outputs=False).apply(
        lambda x: FillNanInf()(Lag(1)(Buffer(look_back)(x)))
    )
    B, T, F = x.shape
    x = x.transpose(1, 0, 2)

    y, _ = dataframe.wax.stream(format_outputs=False).apply(
        lambda x: FillNanInf()(Buffer(look_back)(x))
    )
    y = y.transpose(1, 0, 2)
    return Pair(x, y)


def split_feature_target(
    dataframe,
    look_back=SEQ_LEN,
    stack=True,
    shuffle=False,
    min_periods_ratio: float = 0.8,
) -> Pair:
    x, _ = dataframe.wax.stream(format_outputs=False).apply(
        lambda x: Lag(1)(Buffer(look_back)(x))
    )
    x = x.transpose(1, 0, 2)

    y, _ = dataframe.wax.stream(format_outputs=False).apply(
        lambda x: Buffer(look_back)(x)
    )
    y = y.transpose(1, 0, 2)

    T, B, F = x.shape

    if stack:
        x = x.reshape(T, B * F, 1)
        y = y.reshape(T, B * F, 1)

    if shuffle:
        rng = jax.random.PRNGKey(42)
        idx = jnp.arange(x.shape[1])
        idx = jax.random.shuffle(rng, idx)
        x = x[:, idx]
        y = y[:, idx]

    if min_periods_ratio:
        count_nan = jnp.isnan(x).sum(axis=0)
        mask = count_nan < min_periods_ratio * T
        idx = jnp.where(mask)
        # print("count_nan = ", count_nan)
        # print("B = ", B)
        x = x[:, idx[0], :]
        y = y[:, idx[0], :]
        T, B, F = x.shape
        # print("B = ", B)

    # round Batch size to a power of to
    B_round = int(2 ** jnp.floor(jnp.log2(B)))
    x = x[:, :B_round, :]
    y = y[:, :B_round, :]

    # fillnan by zeros
    fill_nan_inf = hk.transform(lambda x: FillNanInf()(x))
    params = fill_nan_inf.init(None, jnp.full(x.shape, jnp.nan, x.dtype))
    x = fill_nan_inf.apply(params, None, x)
    y = fill_nan_inf.apply(params, None, y)

    return Pair(x, y)
```

```{code-cell} ipython3
def split_train_validation(dataframe, stars, train_size, look_back) -> TrainSplit:

    # prepare scaler
    dataframe_train = dataframe[stars].iloc[:train_size]
    scaler = min_max_scaler(dataframe_train)

    # prepare train data
    dataframe_train_normed = scaler.encode(dataframe_train)
    train = split_feature_target(dataframe_train_normed, look_back)

    # prepare validation data
    valid_size = len(dataframe[stars]) - train_size
    valid_size = int(2 ** jnp.floor(jnp.log2(valid_size)))
    valid_end = int(train_size + valid_size)
    dataframe_valid = dataframe[stars].iloc[train_size:valid_end]
    dataframe_valid_normed = scaler.encode(dataframe_valid)
    valid = split_feature_target(dataframe_valid_normed, look_back)

    return TrainSplit(train, valid)
```

```{code-cell} ipython3
TRAIN_SIZE = 2 ** 16
train, valid = split_train_validation(dataframe_normed, [star], TRAIN_SIZE, SEQ_LEN)
```

```{code-cell} ipython3
train[0].shape, train[1].shape, valid[0].shape, valid[1].shape
```

```{code-cell} ipython3
TRAIN_SIZE, VALID_SIZE = len(train.x), len(valid.x)
```

```{code-cell} ipython3
seq = hk.PRNGSequence(42)
```

```{code-cell} ipython3
# Plot an observation/target pair.
batch_plot = jax.random.choice(next(seq), len(train[0]))
df = pd.DataFrame(
    {"x": train[0][:, batch_plot, 0], "y": train[1][:, batch_plot, 0]}
).reset_index()
df = pd.melt(df, id_vars=["index"], value_vars=["x", "y"])
plot = (
    gg.ggplot(df)
    + gg.aes(x="index", y="value", color="variable")
    + gg.geom_line()
    + gg.scales.scale_y_log10()
)
_ = plot.draw()
```

### Dataset iterator

```{code-cell} ipython3
class Dataset:
    """An iterator over a numpy array, revealing batch_size elements at a time."""

    def __init__(self, xy: Pair, batch_size: int):
        self._x, self._y = xy
        self._batch_size = batch_size
        self._length = self._x.shape[1]
        self._idx = 0
        if self._length % batch_size != 0:
            msg = "dataset size {} must be divisible by batch_size {}."
            raise ValueError(msg.format(self._length, batch_size))

    def __next__(self) -> Pair:
        start = self._idx
        end = start + self._batch_size
        x, y = self._x[:, start:end], self._y[:, start:end]
        if end >= self._length:
            end = end % self._length
            assert end == 0  # Guaranteed by ctor assertion.
        self._idx = end
        return x, y
```

```{code-cell} ipython3
BATCH_SIZE = 8
```

```{code-cell} ipython3
train_ds = Dataset(train, BATCH_SIZE)
valid_ds = Dataset(valid, BATCH_SIZE)
del train, valid  # Don't leak temporaries.
```

+++ {"colab_type": "text", "id": "LZGw5Jdvjmqh"}

### Training an LSTM

To train the LSTM, we define a Haiku function which unrolls the LSTM over the input sequence, generating predictions for all output values. The LSTM always starts with its initial state at the start of the sequence.

The Haiku function is then transformed into a pure function through `hk.transform`, and is trained with Adam on an L2 prediction loss.

```{code-cell} ipython3
from wax.compile import jit_init_apply
```

```{code-cell} ipython3
x, y = next(train_ds)
x.shape, y.shape
```

```{code-cell} ipython3
from collections import defaultdict
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: nacnTj5ejIK5

def unroll_net(seqs: jnp.ndarray):
    """Unrolls an LSTM over seqs, mapping each output to a scalar."""
    # seqs is [T, B, F].
    core = hk.LSTM(32)
    batch_size = seqs.shape[1]
    outs, state = hk.dynamic_unroll(core, seqs, core.initial_state(batch_size))
    # We could include this Linear as part of the recurrent core!
    # However, it's more efficient on modern accelerators to run the linear once
    # over the entire sequence than once per sequence element.
    return hk.BatchApply(hk.Linear(1))(outs), state
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: nacnTj5ejIK5

model = jit_init_apply(hk.transform(unroll_net))
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: nacnTj5ejIK5

def train_model(
    train_ds: Dataset, valid_ds: Dataset, max_iterations: int = -1
) -> hk.Params:
    """Initializes and trains a model on train_ds, returning the final params."""
    rng = jax.random.PRNGKey(428)
    opt = optax.adam(1e-3)

    @jax.jit
    def loss(params, x, y):
        pred, _ = model.apply(params, None, x)
        return jnp.mean(jnp.square(pred - y))

    @jax.jit
    def update(step, params, opt_state, x, y):
        l, grads = jax.value_and_grad(loss)(params, x, y)
        grads, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, grads)
        return l, params, opt_state

    # Initialize state.
    sample_x, _ = next(train_ds)
    params = model.init(rng, sample_x)
    opt_state = opt.init(params)

    step = 0
    records = defaultdict(list)

    def _format_results(records):
        records = {key: jnp.stack(l) for key, l in records.items()}
        return records

    with tqdm() as pbar:
        while True:
            if step % 100 == 0:
                x, y = next(valid_ds)
                valid_loss = loss(params, x, y)
                # print("Step {}: valid loss {}".format(step, valid_loss))
                records["step"].append(step)
                records["valid_loss"].append(valid_loss)

            try:
                x, y = next(train_ds)
            except StopIteration:
                return params, _format_results(records)
            train_loss, params, opt_state = update(step, params, opt_state, x, y)
            if step % 100 == 0:
                # print("Step {}: train loss {}".format(step, train_loss))
                records["train_loss"].append(train_loss)

            step += 1
            pbar.update()
            if max_iterations > 0 and step >= max_iterations:
                return params, _format_results(records)
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: AssgDctokbl5

%%time
trained_params, records = train_model(train_ds, valid_ds, 10000)
```

```{code-cell} ipython3
# Plot losses
losses = pd.DataFrame(records)
df = pd.melt(losses, id_vars=["step"], value_vars=["train_loss", "valid_loss"])
plot = (
    gg.ggplot(df)
    + gg.aes(x="step", y="value", color="variable")
    + gg.geom_line()
    + gg.scales.scale_y_log10()
)
_ = plot.draw()
```

+++ {"colab_type": "text", "id": "yr7jrOL3ki-b"}

### Sampling

The point of training models is so that they can make predictions! How can we generate predictions with the trained model?

If we're allowed to feed in the ground truth, we can just run the original model's `apply` function.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: f2qETEqXLT1N

def plot_samples(truth: np.ndarray, prediction: np.ndarray) -> gg.ggplot:
    assert truth.shape == prediction.shape
    df = pd.DataFrame(
        {"truth": truth.squeeze(), "predicted": prediction.squeeze()}
    ).reset_index()
    df = pd.melt(df, id_vars=["index"], value_vars=["truth", "predicted"])
    plot = (
        gg.ggplot(df) + gg.aes(x="index", y="value", color="variable") + gg.geom_line()
    )
    return plot
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: KOuK1egilGD0

# Grab a sample from the validation set.
sample_x, _ = next(valid_ds)
sample_x = sample_x[:, :1]  # Shrink to batch-size 1.

# Generate a prediction, feeding in ground truth at each point as input.
predicted, _ = model.apply(trained_params, None, sample_x)

plot = plot_samples(sample_x[1:], predicted[:-1])
plot.draw()
del sample_x, predicted
```

### Run autoregressively

+++ {"colab_type": "text", "id": "tDyGshz_lwrM"}

If we can't feed in the ground truth (because we don't have it), we can also run the model autoregressively.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: Cg8oQ75Ulvld

def autoregressive_predict(
    trained_params: hk.Params,
    context: jnp.ndarray,
    seq_len: int,
):
    """Given a context, autoregressively generate the rest of a sine wave."""
    ar_outs = []
    context = jax.device_put(context)
    times = range(seq_len - context.shape[0])
    for _ in times:
        full_context = jnp.concatenate([context] + ar_outs)
        outs, _ = jax.jit(model.apply)(trained_params, None, full_context)
        # Append the newest prediction to ar_outs.
        ar_outs.append(outs[-1:])
    # Return the final full prediction.
    return outs
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: Cg8oQ75Ulvld

sample_x, _ = next(valid_ds)
context_length = SEQ_LEN // 8
# Cut the batch-size 1 context from the start of the sequence.
context = sample_x[:context_length, :1]
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: Cg8oQ75Ulvld

%%time
# We can reuse params we got from training for inference - as long as the
# declaration order is the same.
predicted = autoregressive_predict(trained_params, context, SEQ_LEN)

plot = plot_samples(sample_x[1:, :1], predicted)
plot += gg.geom_vline(xintercept=len(context), linetype="dashed")
plot.draw()
del predicted
```

+++ {"colab_type": "text", "id": "qGkr2gf2oALo"}

#### Sharing parameters with a different function.

Unfortunately, this is a bit slow - we're doing O(N^2) computation for a sequence of length N.

It'd be better if we could do the autoregressive sampling all at once - but we need to write a new Haiku function for that.

We're in luck - if the Haiku module names match, the same parameters can be used for multiple Haiku functions.

This can be achieved through a combination of two techniques:

1. If we manually give a unique name to a module, we can ensure that the parameters are directed to the right places.
2. If modules are instantiated in the same order, they'll have the same names in different functions.

Here, we rely on method #2 to create a fast autoregressive prediction.

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: WdKcHr6_n_ba

def fast_autoregressive_predict_fn(context, seq_len):
    """Given a context, autoregressively generate the rest of a sine wave."""
    core = hk.LSTM(32)
    dense = hk.Linear(1)
    state = core.initial_state(context.shape[1])
    # Unroll over the context using `hk.dynamic_unroll`.
    # As before, we `hk.BatchApply` the Linear for efficiency.
    context_outs, state = hk.dynamic_unroll(core, context, state)
    context_outs = hk.BatchApply(dense)(context_outs)

    # Now, unroll one step at a time using the running recurrent state.
    ar_outs = []
    x = context_outs[-1]
    times = range(seq_len - context.shape[0])
    for _ in times:
        x, state = core(x, state)
        x = dense(x)
        ar_outs.append(x)
    return jnp.concatenate([context_outs, jnp.stack(ar_outs)])


fast_ar_predict = hk.transform(fast_autoregressive_predict_fn)
fast_ar_predict = jax.jit(fast_ar_predict.apply, static_argnums=3)
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: WdKcHr6_n_ba

%%time
# Reuse the same context from the previous cell.
predicted = fast_ar_predict(trained_params, None, context, SEQ_LEN)

# The plots should be equivalent!
plot = plot_samples(sample_x[1:, :1], predicted[:-1])
plot += gg.geom_vline(xintercept=len(context), linetype="dashed")
_ = plot.draw()
```

```{code-cell} ipython3
:colab: {}
:colab_type: code
:id: 9S0tkPXGrU3a

%timeit autoregressive_predict(trained_params, context, SEQ_LEN)
%timeit fast_ar_predict(trained_params, None, context, SEQ_LEN)
```

## Train all stars

+++

### Training

```{code-cell} ipython3
def split_train_validation_date(dataframe, stars, date, look_back) -> TrainSplit:
    train_size = len(dataframe.loc[:date])
    return split_train_validation(dataframe, stars, train_size, look_back)
```

```{code-cell} ipython3
%%time
TRAIN_DATE = "2016"
train, valid = split_train_validation_date(dataframe_normed, stars, TRAIN_DATE, SEQ_LEN)
TRAIN_SIZE = train[0].shape[1]
print(f"TRAIN_SIZE = {TRAIN_SIZE}")
```

```{code-cell} ipython3
train[0].shape, train[1].shape, valid[0].shape, valid[1].shape
```

```{code-cell} ipython3
train_ds = Dataset(train, BATCH_SIZE)
valid_ds = Dataset(valid, BATCH_SIZE)
del train, valid  # Don't leak temporaries.
```

```{code-cell} ipython3
%%time
trained_params, records = train_model(train_ds, valid_ds, 10000)
```

```{code-cell} ipython3
# Plot losses
losses = pd.DataFrame(records)
df = pd.melt(losses, id_vars=["step"], value_vars=["train_loss", "valid_loss"])
plot = (
    gg.ggplot(df)
    + gg.aes(x="step", y="value", color="variable")
    + gg.geom_line()
    + gg.scales.scale_y_log10()
)
_ = plot.draw()
```

### Sampling

```{code-cell} ipython3
# Grab a sample from the validation set.
sample_x, _ = next(valid_ds)
sample_x = sample_x[:, :1]  # Shrink to batch-size 1.

# Generate a prediction, feeding in ground truth at each point as input.
predicted, _ = model.apply(trained_params, None, sample_x)

plot = plot_samples(sample_x[1:], predicted[:-1])
plot.draw()
del sample_x, predicted
```

### Run autoregressively

```{code-cell} ipython3
%%time
sample_x, _ = next(valid_ds)
context_length = SEQ_LEN // 8
# Cut the batch-size 1 context from the start of the sequence.
context = sample_x[:context_length, :1]

# Reuse the same context from the previous cell.
predicted = fast_ar_predict(trained_params, None, context, SEQ_LEN)

# The plots should be equivalent!
plot = plot_samples(sample_x[1:, :1], predicted[:-1])
plot += gg.geom_vline(xintercept=len(context), linetype="dashed")
_ = plot.draw()
```
