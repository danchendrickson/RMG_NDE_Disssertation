# %% [markdown]
# # Quick Start Tutorial
# 
# GluonTS contains:
# 
# * A number of pre-built models
# * Components for building new models (likelihoods, feature processing pipelines, calendar features etc.)
# * Data loading and processing
# * Plotting and evaluation facilities
# * Artificial and real datasets (only external datasets with blessed license)

# %%
# %%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# %% [markdown]
# ## Datasets
# 
# ### Provided datasets
# 
# GluonTS comes with a number of publicly available datasets.

# %%
from gluonts.dataset.repository import get_dataset, dataset_names
from gluonts.dataset.util import to_pandas

# %%
print(f"Available datasets: {dataset_names}")

# %% [markdown]
# To download one of the built-in datasets, simply call get_dataset with one of the above names. GluonTS can re-use the saved dataset so that it does not need to be downloaded again the next time around.

# %%
dataset = get_dataset("m4_hourly")

# %% [markdown]
# In general, the datasets provided by GluonTS are objects that consists of three main members:
# 
# - `dataset.train` is an iterable collection of data entries used for training. Each entry corresponds to one time series.
# - `dataset.test` is an iterable collection of data entries used for inference. The test dataset is an extended version of the train dataset that contains a window in the end of each time series that was not seen during training. This window has length equal to the recommended prediction length.
# - `dataset.metadata` contains metadata of the dataset such as the frequency of the time series, a recommended prediction horizon, associated features, etc.

# %%
entry = next(iter(dataset.train))
train_series = to_pandas(entry)
train_series.plot()
plt.grid(which="both")
plt.legend(["train series"], loc="upper left")
plt.savefig("TraingSeries.png")
plt.show()

# %%
entry = next(iter(dataset.test))
test_series = to_pandas(entry)
test_series.plot()
plt.axvline(train_series.index[-1], color="r")  # end of train dataset
plt.grid(which="both")
plt.legend(["test series", "end of train series"], loc="upper left")
plt.savefig("TestTrainSeries.png")
plt.show()

# %%
print(
    f"Length of forecasting window in test dataset: {len(test_series) - len(train_series)}"
)
print(f"Recommended prediction horizon: {dataset.metadata.prediction_length}")
print(f"Frequency of the time series: {dataset.metadata.freq}")

# %% [markdown]
# ### Custom datasets
# 
# At this point, it is important to emphasize that GluonTS does not require this specific format for a custom dataset that a user may have. The only requirements for a custom dataset are to be iterable and have a "target" and a "start" field. To make this more clear, assume the common case where a dataset is in the form of a `numpy.array` and the index of the time series in a `pandas.Period` (possibly different for each time series):

# %%
N = 10  # number of time series
T = 100  # number of timesteps
prediction_length = 24
freq = "1H"
custom_dataset = np.random.normal(size=(N, T))
start = pd.Period("01-01-2019", freq=freq)  # can be different for each time series

# %% [markdown]
# Now, you can split your dataset and bring it in a GluonTS appropriate format with just two lines of code:

# %%
from gluonts.dataset.common import ListDataset

# %%
# train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
train_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset[:, :-prediction_length]],
    freq=freq,
)
# test dataset: use the whole dataset, add "target" and "start" fields
test_ds = ListDataset(
    [{"target": x, "start": start} for x in custom_dataset], freq=freq
)

# %% [markdown]
# ## Training an existing model (`Estimator`)
# 
# GluonTS comes with a number of pre-built models. All the user needs to do is configure some hyperparameters. The existing models focus on (but are not limited to) probabilistic forecasting. Probabilistic forecasts are predictions in the form of a probability distribution, rather than simply a single point estimate.
# 
# We will begin with GluonTS's pre-built feedforward neural network estimator, a simple but powerful forecasting model. We will use this model to demonstrate the process of training a model, producing forecasts, and evaluating the results.
# 
# GluonTS's built-in feedforward neural network (`SimpleFeedForwardEstimator`) accepts an input window of length `context_length` and predicts the distribution of the values of the subsequent `prediction_length` values. In GluonTS parlance, the feedforward neural network model is an example of an `Estimator`. In GluonTS, `Estimator` objects represent a forecasting model as well as details such as its coefficients, weights, etc.
# 
# In general, each estimator (pre-built or custom) is configured by a number of hyperparameters that can be either common (but not binding) among all estimators (e.g., the `prediction_length`) or specific for the particular estimator (e.g., number of layers for a neural network or the stride in a CNN).
# 
# Finally, each estimator is configured by a `Trainer`, which defines how the model will be trained i.e., the number of epochs, the learning rate, etc.

# %%
from gluonts.mx import SimpleFeedForwardEstimator, Trainer

# %%
estimator = SimpleFeedForwardEstimator(
    num_hidden_dimensions=[10],
    prediction_length=dataset.metadata.prediction_length,
    context_length=100,
    trainer=Trainer(ctx="cpu", epochs=5, learning_rate=1e-3, num_batches_per_epoch=100),
)

# %% [markdown]
# After specifying our estimator with all the necessary hyperparameters we can train it using our training dataset `dataset.train` by invoking the `train` method of the estimator. The training algorithm returns a fitted model (or a `Predictor` in GluonTS parlance) that can be used to construct forecasts.

# %%
predictor = estimator.train(dataset.train)

# %% [markdown]
# ## Visualize and evaluate forecasts
# 
# With a predictor in hand, we can now predict the last window of the `dataset.test` and evaluate our model's performance.
# 
# GluonTS comes with the `make_evaluation_predictions` function that automates the process of prediction and model evaluation. Roughly, this function performs the following steps:
# 
# - Removes the final window of length `prediction_length` of the `dataset.test` that we want to predict
# - The estimator uses the remaining data to predict (in the form of sample paths) the "future" window that was just removed
# - The module outputs the forecast sample paths and the `dataset.test` (as python generator objects)

# %%
from gluonts.evaluation import make_evaluation_predictions

# %%
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,  # test dataset
    predictor=predictor,  # predictor
    num_samples=100,  # number of sample paths we want for evaluation
)

# %% [markdown]
# First, we can convert these generators to lists to ease the subsequent computations.

# %%
forecasts = list(forecast_it)
tss = list(ts_it)

# %% [markdown]
# We can examine the first element of these lists (that corresponds to the first time series of the dataset). Let's start with the list containing the time series, i.e., `tss`. We expect the first entry of `tss` to contain the (target of the) first time series of `dataset.test`.

# %%
# first entry of the time series list
ts_entry = tss[0]

# %%
# first 5 values of the time series (convert from pandas to numpy)
np.array(ts_entry[:5]).reshape(
    -1,
)

# %%
# first entry of dataset.test
dataset_test_entry = next(iter(dataset.test))

# %%
# first 5 values
dataset_test_entry["target"][:5]

# %% [markdown]
# The entries in the `forecast` list are a bit more complex. They are objects that contain all the sample paths in the form of `numpy.ndarray` with dimension `(num_samples, prediction_length)`, the start date of the forecast, the frequency of the time series, etc. We can access all this information by simply invoking the corresponding attribute of the forecast object.

# %%
# first entry of the forecast list
forecast_entry = forecasts[0]

# %%
print(f"Number of sample paths: {forecast_entry.num_samples}")
print(f"Dimension of samples: {forecast_entry.samples.shape}")
print(f"Start date of the forecast window: {forecast_entry.start_date}")
print(f"Frequency of the time series: {forecast_entry.freq}")

# %% [markdown]
# We can also do calculations to summarize the sample paths, such as computing the mean or a quantile for each of the 48 time steps in the forecast window.

# %%
print(f"Mean of the future window:\n {forecast_entry.mean}")
print(f"0.5-quantile (median) of the future window:\n {forecast_entry.quantile(0.5)}")

# %% [markdown]
# `Forecast` objects have a `plot` method that can summarize the forecast paths as the mean, prediction intervals, etc. The prediction intervals are shaded in different colors as a "fan chart".

# %%
plt.plot(ts_entry[-150:].to_timestamp())
forecast_entry.plot(show_label=True)
plt.legend()
plt.savefig("entryPlot.png")

# %% [markdown]
# We can also evaluate the quality of our forecasts numerically. In GluonTS, the `Evaluator` class can compute aggregate performance metrics, as well as metrics per time series (which can be useful for analyzing performance across heterogeneous time series).

# %%
from gluonts.evaluation import Evaluator

# %%
evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(tss, forecasts)

# %% [markdown]
# The aggregate metrics, `agg_metrics`, aggregate both across time-steps and across time series.

# %%
print(json.dumps(agg_metrics, indent=4))

# %% [markdown]
# Individual metrics are aggregated only across time-steps.

# %%
item_metrics.head()

# %%
item_metrics.plot(x="MSIS", y="MASE", kind="scatter")
plt.grid(which="both")
plt.savefig("MsisMase.png")
plt.show()


