# %% [markdown]
# # Timeseries anomaly detection using an Autoencoder
# 
# **Author:** [pavithrasv](https://github.com/pavithrasv)<br>
# **Date created:** 2020/05/31<br>
# **Last modified:** 2020/05/31<br>
# **Description:** Detect anomalies in a timeseries using an Autoencoder.

# %% [markdown]
# https://github.com/keras-team/keras-io/blob/master/examples/timeseries/ipynb/timeseries_anomaly_detection.ipynb

# %% [markdown]
# ## Introduction
# 
# This script demonstrates how you can use a reconstruction convolutional
# autoencoder model to detect anomalies in timeseries data.

# %% [markdown]
# ## Setup

# %%
import numpy as np
import pandas as pd
import keras
from keras import layers, models
from matplotlib import pyplot as plt

# %%
from joblib import delayed

# %%
import pandas as pd

# %%
import random

# %%
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

# %%
import dask.dataframe as dd
import os
import shutil

# %%
batchSize = 256

# %% [markdown]
# ## Load the data
# 
# We will use the [Numenta Anomaly Benchmark(NAB)](
# https://www.kaggle.com/boltzmannbrain/nab) dataset. It provides artificial
# timeseries data containing labeled anomalous periods of behavior. Data are
# ordered, timestamped, single-valued metrics.
# 
# We will use the `art_daily_small_noise.csv` file for training and the
# `art_daily_jumpsup.csv` file for testing. The simplicity of this dataset
# allows us to demonstrate anomaly detection effectively.

# %% [markdown]
# ## Build a model
# 
# We will build a convolutional reconstruction autoencoder model. The model will
# take input of shape `(batch_size, sequence_length, num_features)` and return
# output of the same shape. In this case, `sequence_length` is 288 and
# `num_features` is 1.

# %% [markdown]
# ### Get Data  sequences
# Create sequences combining `TIME_STEPS` contiguous data values from the
# training data.

# %%
# %%time
TIME_STEPS = 1000
Dims = 3

Folder = '/scratch/1000Sm/'
Folder = '/scratch/1000Input/'
Folder = '/lclscr/1000Inputs/'

# %% [markdown]
# %%time 
Olines = [
    os.path.join(Folder,file)
    for file in os.listdir(Folder) if file.endswith('Outs.csv') #and file.startswith('2')
]

# %%
random.shuffle(Olines)

Quarter = int(len(Olines)/4)
lines = [sub.replace('Outs', 'Data') for sub in Olines[:3*Quarter]]
TestLines = [sub.replace('Outs', 'Data') for sub in Olines[3*Quarter:]]


#with open('FileListAsOf0812-b.txt', 'r') as file:
#    # Read all lines into a list
#    lines = file.readlines()

# Optionally, strip newline characters from each line
#lines = [line.strip() for line in lines]

# Print the list to verify
#print(lines)

# %%
len(lines)

# %% [markdown]
# dataset = tf.data.Dataset.list_files(file_list)

# %%
def parse_csv(file_path):
    # Read the CSV file
    try:
        df = pd.read_csv(file_path, header=None)
        df.columns = ['rx','ry','rz']
        features = df[['rx','ry','rz']].values.tolist()
    except:
        features = np.zeros((1000,3)).tolist()
    
    try:
        label = pd.read_csv(file_path[:-8]+'Outs.csv', header=None)
        label.columns = ['sx']
        labels = np.asarray(label.sx)
    except:
        labels = np.asarray(np.zeros(1000,1))
        
    return features, labels

# %%
class CustomDataset(tf.keras.utils.Sequence):
    def __init__(self, files, batch_size=batchSize, shuffle=True, **kwargs):
        super().__init__(**kwargs)  # Call the parent class constructor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.files = files
        self.on_epoch_end()
      
    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        Feat, Labe = [], []
        
        for idx in indexes:
            f = parse_csv(self.files[idx])
            Feat.append(f[0])
            Labe.append(f[1])

        Feat = tf.convert_to_tensor(Feat, dtype=tf.float32)
        Labe = tf.convert_to_tensor(Labe, dtype=tf.float32)
        return Feat, Labe

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle:
            np.random.shuffle(self.indexes)


# %%
training_generator = CustomDataset(lines, batch_size=batchSize, shuffle = True)

# %% [markdown]
# ## Train the model
# 
# Please note that we are using `x_train` as both the input and the target
# since this is a reconstruction model.

# %%
model = models.Sequential(
    [
        # Input layer
        layers.Input(shape=(1000, 3)),
        
        # Increased number of filters and added layers
        layers.Conv1D(filters=128, kernel_size=7, padding="same", strides=2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Conv1D(filters=64, kernel_size=7, padding="same", strides=2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Conv1D(filters=32, kernel_size=7, padding="same", strides=2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Dropout layer for regularization
        layers.Dropout(rate=0.3),
        
        # Transpose layers with increased filters
        layers.Conv1DTranspose(filters=32, kernel_size=7, padding="same", strides=2, activation="relu"),
        layers.Conv1DTranspose(filters=64, kernel_size=7, padding="same", strides=2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        layers.Conv1DTranspose(filters=128, kernel_size=7, padding="same", strides=2, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        
        # Dropout layer for regularization
        layers.Dropout(rate=0.3),
        
        # Output layer
        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
        
        # Added dense layers with more neurons
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1)
    ]
)

# %%
class CustomModelCheckpoint(Callback):
    def __init__(self, filepath, save_freq):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.save_freq = save_freq

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            self.model.save(self.filepath.format(epoch=epoch + 1), save_format='keras')


# %%
checkpoint_callback = CustomModelCheckpoint(
    filepath='/scratch/models/TAD_0823_checkpoint_B523_{epoch:02d}.keras',
    save_freq=1  
)

tb_callback = tf.keras.callbacks.TensorBoard(log_dir='/scratch/models/profiles/0823-512/',
                                            profile_batch='01, 256')

es_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=5, mode="min")

# %%
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse")
model.summary()

# %%
with tf.device('/gpu:0'):
    history = model.fit(
        x=training_generator,
        epochs=5,
        batch_size=batchSize,
        #validation_split=0.1,
        callbacks=[checkpoint_callback, es_callback, tb_callback], 
    )

# %% [markdown]
# Let's plot training and validation loss to see how the training went.

# %%
plt.plot(history.history["loss"], label="Training Loss")
#plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.savefig('LearningLoss-0827.png')
plt.show()

# %%
