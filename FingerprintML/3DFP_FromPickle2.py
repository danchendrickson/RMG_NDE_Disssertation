#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import os as os
from time import time as ti
from time import ctime as ct
from time import sleep
from skimage.restoration import denoise_wavelet
import pickle
import sys
import random
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import psutil
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

import numpy as np
import argparse
from tensorflow.keras.callbacks import Callback
import keras
import re
import random
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from datetime import datetime as dt

DataFolder = '/scratch/Recordings2/MLPickles'

try:
    LastGoodModel = int(sys.argv[1])
except:
    LastGoodModel = 0
try:
    LastSuccesfull = int(sys.argv[2])
except:
    LastSuccesfull = 0

try:
    DateString = sys.argv[3]
except:
    DateString = '1114'

tic = ti()
start = tic

MemoryProtection = True

LR_Starting = 2e-2
LR_Current = LR_Starting
LR_PeriodGrow = 6
LR_Decay = .75
LR_Expand = 3.75



# initialize the number of epochs to train for and batch size
EPOCHS = 25
BS = 64
TestSplit = 10 # 1/this many


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())

#tf.compat.v1.disable_eager_execution()

sleep(5)

if LastSuccesfull == 0:
    files= os.listdir(DataFolder) 
    random.shuffle(files)
    with open(f'PickleList{DateString}.text','w') as file:
        for item in files:
            file.write(f"{item}\n")
else:
    with open(f'PickleList{DateString}.text','r') as file:
        files = file.readlines()
    files=[item.strip() for item in files]

print('got files')
# cite: https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/

class ConvAutoencoder:
    @staticmethod
    def build(width, height, depth, filters=(32, 64), latentDim=24):
        inputShape = (height, width, depth)
        chanDim = -1
        inputs = Input(shape=inputShape)
        x = inputs

        for f in filters:
            x = Conv2D(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        volumeSize = K.int_shape(x)
        print("Volume Size:", volumeSize)
        x = Flatten()(x)
        latent = Dense(latentDim)(x)

        encoder = Model(inputs, latent, name="encoder")

        latentInputs = Input(shape=(latentDim,))
        flattenedVolumeSize = int(np.prod(volumeSize[1:]))
        print("Flattened Volume Size:", flattenedVolumeSize)
        x = Dense(flattenedVolumeSize)(latentInputs)
        x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

        for f in filters[::-1]:
            x = Conv2DTranspose(f, (3, 3), strides=2, padding="same")(x)
            x = LeakyReLU(alpha=0.2)(x)
            x = BatchNormalization(axis=chanDim)(x)

        x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
        outputs = Activation("sigmoid")(x)

        decoder = Model(latentInputs, outputs, name="decoder")
        autoencoder = Model(inputs, decoder(encoder(inputs)), name="autoencoder")

        return (encoder, decoder, autoencoder)


ImageShape=[1, 32, 600, 3]

toc=ti()


DataFolder = '/scratch/Recordings2/MLPickles'

# Setup TensorFlow to use GPU memory growth
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Load model and define optimizer
if LastGoodModel == 0:
    (encoder, decoder, autoencoder) = ConvAutoencoder.build(ImageShape[2], ImageShape[1], ImageShape[3],(64,32),32)
else:

    directory = '/scratch/models/'

    # Regular expression to match the filenames
    pattern = re.compile(r'3DFP_(\d{4})_(\d{3})-autoencoder\.keras')
    
    # Initialize variables to track the highest numbers
    max_main_number = -1
    max_sub_number = -1
    target_file = None
    
    target_file = f'3DFP_{DateString}_{str(LastGoodModel).zfill(3)}-autoencoder.keras'
    encoder = load_model(directory+target_file[:-18]+'-encoder.keras')
    decoder = load_model(directory+target_file[:-18]+'-decoder.keras')

    autoencoder_input = Input(shape=(ImageShape[1], ImageShape[2], ImageShape[3]))

    # Pass the input through the encoder and decoder
    encoded_repr = encoder(autoencoder_input)
    reconstructed = decoder(encoded_repr)

    # Create the reassembled autoencoder model
    autoencoder = Model(autoencoder_input, reconstructed)

opt = Adam(learning_rate=LR_Starting)
autoencoder.compile(loss="mse", optimizer=opt)

def load_large_tensor_in_chunks(file_path, chunk_size):
    with open(file_path, 'rb') as f:
        while True:
            try:
                chunk = pickle.load(f, encoding='latin1')
                yield np.array(chunk, dtype=np.float32)  # Ensure correct data type for TensorFlow
            except EOFError:
                break

def load_large_tensor_in_chunks(file_path, chunk_size):
    with tf.device('/cpu:0'):
        with open(file_path, 'rb') as f:
            tenTemp = pickle.load(f)
            tensor = tenTemp.numpy()
            del tenTemp
    chunks = [tensor[i:i + chunk_size] for i in range(0, tensor.shape[0], chunk_size)]
    return chunks

def data_generator(chunks, batch_size):
    def generator():
        for chunk in chunks:
            num_samples = chunk.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]
                batch_tensor = chunk[batch_indices]
                print(f"Generated batch with shape: {batch_tensor.shape}")
                yield batch_tensor.astype(np.float32)  # Ensure correct data type for TensorFlow
    return generator

def create_dataset(generator, batch_size):
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(batch_size, 32, 600, 3), dtype=tf.float32)
    ).batch(batch_size)
    
    # Prefetch to GPU
    dataset = dataset.apply(tf.data.experimental.prefetch_to_device('/GPU:0', buffer_size=batch_size))
    return dataset

def split_dataset(dataset, validation_split=0.1):
    dataset_size = len(list(dataset.as_numpy_iterator()))
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)

    return train_dataset, val_dataset


# Training loop
LoopsToGetAll = len(files) - LastSuccesfull
print(f'Loops Needed: {LoopsToGetAll} at time {ct(ti())}.')

opt = Adam(learning_rate=LR_Starting)
autoencoder.compile(loss="mse", optimizer=opt)

for j in range(LoopsToGetAll):
    j += 1 + LastSuccesfull

    large_tensor_chunks = load_large_tensor_in_chunks(os.path.join(DataFolder, files[j]), chunk_size=2*BS)
    print(f'File Opened {files[j]}, processing in chunks.')

    generator = data_generator(large_tensor_chunks, BS)
    dataset = create_dataset(generator, BS)

    train_dataset, val_dataset = split_dataset(dataset)

    H = autoencoder.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        steps_per_epoch=len(list(train_dataset.as_numpy_iterator())) // BS,
        validation_steps=len(list(val_dataset.as_numpy_iterator())) // BS
    )

    autoencoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-autoencoder.keras')
    encoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-encoder.keras')
    decoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-decoder.keras')
    print(f'saved 3DFP_{DateString}_{str(j).zfill(3)}-autoencoder')

    if j % LR_PeriodGrow == 0:
        LR_Current *= LR_Expand
    else:
        LR_Current *= LR_Decay

    autoencoder.optimizer.learning_rate = LR_Current

    print(f'{j} of {LoopsToGetAll + LastSuccesfull + 1} in {int((ti() - toc) / .6) / 100} minutes at time {ct(ti())}. Using {psutil.virtual_memory()[2]} of RAM')


'''

for j in range(LoopsToGetAll):
    j += 1 + LastSuccesfull

    large_tensor_chunks = load_large_tensor_in_chunks(os.path.join(DataFolder, files[j]), chunk_size=1024)
    print(f'File Opened {files[j]}, processing in chunks.')

    generator = data_generator(large_tensor_chunks, BS)
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=tf.TensorSpec(shape=(None, 32, 600, 3), dtype=tf.float32)
    ).batch(BS)

    train_dataset, val_dataset = split_dataset(dataset)

    H = autoencoder.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        steps_per_epoch=len(list(train_dataset.as_numpy_iterator())) // BS,
        validation_steps=len(list(val_dataset.as_numpy_iterator())) // BS
    )

    autoencoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-autoencoder.keras')
    encoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-encoder.keras')
    decoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-decoder.keras')
    print(f'saved 3DFP_{DateString}_{str(j).zfill(3)}-autoencoder')

    if j % LR_PeriodGrow == 0:
        LR_Current *= LR_Expand
    else:
        LR_Current *= LR_Decay

    autoencoder.optimizer.learning_rate = LR_Current

    print(f'{j} of {LoopsToGetAll + LastSuccesfull + 1} in {int((ti() - toc) / .6) / 100} minutes at time {ct(ti())}. Using {psutil.virtual_memory()[2]} of RAM')

for j in range(LoopsToGetAll):
    j += 1 + LastSuccesfull

    large_tensor = load_large_tensor(os.path.join(DataFolder, files[j]))
    ImageShape = large_tensor.shape
    print(f'File Opened {files[j]}, Shape is {ImageShape}')

    if ImageShape[0] > 100:
        generator = data_generator(large_tensor, BS)
        dataset = create_dataset(generator, BS)

        # Split the dataset into training and validation datasets
        train_dataset, val_dataset = split_dataset(dataset)

        H = autoencoder.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            steps_per_epoch=len(large_tensor) // BS,
            validation_steps=len(list(val_dataset.as_numpy_iterator())) // BS
        )

        autoencoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-autoencoder.keras')
        encoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-encoder.keras')
        decoder.save(f'/scratch/models/3DFP_{DateString}_{str(j).zfill(3)}-decoder.keras')
        print(f'saved 3DFP_{DateString}_{str(j).zfill(3)}-autoencoder')
        
        if j % LR_PeriodGrow == 0:
            LR_Current *= LR_Expand
        else:
            LR_Current *= LR_Decay
            
        autoencoder.optimizer.learning_rate = LR_Current

    # Free memory
    del large_tensor
    tf.keras.backend.clear_session()
    gc.collect()

    print(f'{j} of {LoopsToGetAll + LastSuccesfull + 1} in {int((ti() - toc) / .6) / 100} minutes at time {ct(ti())}. Using {psutil.virtual_memory()[2]} of RAM')
'''