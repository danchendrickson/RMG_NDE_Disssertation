#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import os as os
import multiprocessing
from joblib import Parallel, delayed
from time import time as ti
from time import ctime as ct
from skimage.restoration import denoise_wavelet
import pickle
#import CoreFunctions as cf
import sys
import random
import psutil


# In[2]:


# set the matplotlib backend so figures can be saved in the background
#import matplotlib
#matplotlib.use("Agg")
# import the necessary packages
#from pyimagesearch.convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import argparse
#import cv2
# construct the argument parse and parse the arguments


# In[3]:


from tensorflow.keras.callbacks import Callback


# In[4]:


import keras
import re


# In[5]:


import random
import tensorflow as tf
from tensorflow.keras.models import load_model


# In[6]:


# import the necessary packages
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
import numpy as np


# In[7]:


DataFolder = '/scratch/Recordings2/MLPickles'

LastGoodModel = 2

LastSuccesfull = 2

DateString = '2222'

tic = ti()
start = tic

MemoryProtection = True

LR_Starting = 2e-2
LR_Current = LR_Starting
LR_PeriodGrow = 6
LR_Decay = .75
LR_Expand = 3.75


# In[8]:


# initialize the number of epochs to train for and batch size
EPOCHS = 25
BS = 128
TestSplit = 10 # 1/this many


# In[9]:


from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

print(get_available_gpus())


# In[10]:


#os.environ['CUDA_VISIBLE_DEVICES'] = '2'


# In[11]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


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


# # Start Machine Learning
# ## Using Autoencoder with Kears and Tensorflow
# cite: https://pyimagesearch.com/2020/02/17/autoencoders-with-keras-tensorflow-and-deep-learning/

# In[13]:


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


# In[14]:


#with open(DataFolder + f'/{files[1]}', 'rb') as file:
#    trX = pickle.load(file)
#ImageShape = trX.shape


# In[15]:


ImageShape=[43495, 32, 600, 3]


# In[16]:


if LastGoodModel == 0:
    (encoder, decoder, autoencoder) = ConvAutoencoder.build(ImageShape[2], ImageShape[1], ImageShape[3],(128,64),32)
else:

    directory = '/scratch/models/'

    # Regular expression to match the filenames
    pattern = re.compile(r'3DFP_(\d{4})_(\d{3})-autoencoder\.keras')
    
    # Initialize variables to track the highest numbers
    max_main_number = -1
    max_sub_number = -1
    target_file = None
    
    target_file = f'3DFP_{DateString}_{str(LastGoodModel).zfill(3)}-autoencoder.keras'
    reautoencoder = load_model(directory+target_file)
    encoder = load_model(directory+target_file[:-18]+'-encoder.keras')
    decoder = load_model(directory+target_file[:-18]+'-decoder.keras')

    autoencoder_input = Input(shape=(ImageShape[1], ImageShape[2], ImageShape[3]))

    # Pass the input through the encoder and decoder
    encoded_repr = encoder(autoencoder_input)
    reconstructed = decoder(encoded_repr)

    # Create the reassembled autoencoder model
    autoencoder = Model(autoencoder_input, reconstructed)


# In[17]:


print(encoder.summary())
print(decoder.summary())
print(autoencoder.summary())
if LastSuccesfull != 0:
    print(reautoencoder.summary())


# In[18]:


opt = Adam(learning_rate=LR_Starting)
autoencoder.compile(loss="mse", optimizer=opt)
# train the convolutional autoencoder


# In[19]:


toc=ti()


# In[ ]:


LoopsToGetAll = int(len(files))-LastSuccesfull
print(f'Loops Needed: {LoopsToGetAll} at time {ct(ti())}.')
for j in range(LoopsToGetAll):
    j+=1+LastSuccesfull
    with open(DataFolder + f'/{files[j]}', 'rb') as file:
        trX = pickle.load(file)
    ImageShape = trX.shape
    print(f'File Opened {files[j]}, Shape is {ImageShape}')
    if ImageShape[0] > 100:
            
        H = autoencoder.fit(
            trX, trX,
            validation_split=0.1,
            epochs=20,
            #callbacks=[checkpoint_callback, es_callback],     
            batch_size=BS)
        
        plt.plot(H.history["loss"], label="Training Loss")
        plt.plot(H.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.show()
    
        #random.shuffle(trX)
        autoencoder.save('/scratch/models/3DFP_'+DateString+'_'+str(j).zfill(3)+'-autoencoder.keras')
        encoder.save('/scratch/models/3DFP_'+DateString+'_'+str(j).zfill(3)+'-encoder.keras')
        decoder.save('/scratch/models/3DFP_'+DateString+'_'+str(j).zfill(3)+'-decoder.keras')
        print('saved 3DFP_'+DateString+'_'+str(j).zfill(3)+'-autoencoder')
        
        x_train_pred = autoencoder.predict(trX[:7])
        
        for i in range(7):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(12,2), dpi=200 )
            ax1.imshow(trX[i], origin='lower',aspect='auto')
            ax1.axis("off")
            ax2.imshow(x_train_pred[i], origin='lower',aspect='auto')
            ax2.axis("off")
            ax3.imshow(np.abs(trX[i]-x_train_pred[i]), origin='lower',aspect='auto')
            ax3.axis("off")
            plt.show()
    
        del trX, x_train_pred, fig, ax1,ax2,ax3
    
        if j%LR_PeriodGrow == 0:
            LR_Current *= LR_Expand
        else:
            LR_Current *= LR_Decay
            
        autoencoder.optimizer.learning_rate = LR_Current

    print(f'{j} of {LoopsToGetAll+LastSuccesfull+1} in {int((ti()-toc)/.6)/100} minutes at time {ct(ti())}. Using { psutil.virtual_memory()[2]} of RAM')


# In[ ]:


get_ipython().run_line_magic('whos', '')


# In[ ]:


test

