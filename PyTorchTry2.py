# %% [markdown]
# # Trying PyTorch for LSTM Encoder / Decoder
# borrowed from interwebs
# https://github.com/lkulowski/LSTM_encoder_decoder

# %% [markdown]
# ## Standard header

# %%
#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.special as sp

#Custome graph format style sheet
#plt.style.use('Prospectus.mplstyle')

#If being run by a seperate file, use the seperate file's graph format and saving paramaeters
#otherwise set what is needed
if not 'Saving' in locals():
    Saving = False
if not 'Titles' in locals():
    Titles = True
if not 'Ledgends' in locals():
    Ledgends = True
if not 'FFormat' in locals():
    FFormat = '.png'

#Standard cycle to make black and white images and dashed and line styles
default_cycler = (cycler('color', ['0.00', '0.40', '0.60', '0.70']) + cycler(linestyle=['-', '-', '-', '-']))
plt.rc('axes', prop_cycle=default_cycler)
my_cmap = plt.get_cmap('gray')

# %%
#Extra Headers:
import os as os
import pywt as py
import statistics as st
import os as os
import random
import multiprocessing
from joblib import Parallel, delayed
import platform

from time import time as ti

# %%
#import CoreFunctions as cf
from skimage.restoration import denoise_wavelet

# %% [markdown]
# ## Choosing platfrom and folders

# %%
HostName = platform.node()

if HostName == "Server":
    Computer = "Desktop"   
elif HostName[-6:] == 'wm.edu':
    Computer = "SciClone"
elif HostName == "SchoolLaptop":
    Computer = "LinLap"
elif HostName == "WTC-TAB-512":
    Computer = "PortLap"
else:
    Computer = "WinLap"

# %%
if Computer == "SciClone":
    location = '/sciclone/home20/dchendrickson01/image/'
elif Computer == "WinLap":
    location = 'C:\\Data\\'
elif Computer == "Desktop":
    location = "E:\\Backups\\Dan\\CraneData\\"
elif Computer == "LinLap":
    location = '/home/dan/Output/'
elif Computer == 'PortLap':
    location = 'C:\\users\\dhendrickson\\Desktop\\AccelData\\'

# %%
if Computer ==  "SciClone":
    rootfolder = '/sciclone/home20/dchendrickson01/'
    folder = '/sciclone/scr10/dchendrickson01/Recordings2/'
    imageFolder = '/sciclone/scr10/dchendrickson01/Move3Dprint/'
elif Computer == "Desktop":
    rootfolder = location
    folder = rootfolder + "Recordings2\\"
elif Computer =="WinLap":
    rootfolder = location
    folder = rootfolder + "Recordings2\\"   
elif Computer == "LinLap":
    rootfolder = '/home/dan/Data/'
    folder = rootfolder + 'Recordings2/'
elif Computer =='PortLap':
    rootfolder = location 
    folder = rootfolder + 'Recordings2\\'

# %% [markdown]
# ## Start Code

# %%
# Author: Laura Kulowski

'''
Example of using a LSTM encoder-decoder to model a synthetic time series 
https://github.com/lkulowski/LSTM_encoder_decoder

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload
import sys

import lstm_encoder_decoder
import plotting

matplotlib.rcParams.update({'font.size': 17})

# %%
import torch

# %%
def train_test_split(t, y, split = 0.8):

    '''
    
    split time series into train/test sets
    
    : param t:                      time array
    : para y:                       feature array
    : para split:                   percent of data to include in training set 
    : return t_train, y_train:      time/feature training and test sets;  
    :        t_test, y_test:        (shape: [# samples, 1])
    
    '''
    
    indx_split = int(split * len(y))
    indx_train = np.arange(0, indx_split)
    indx_test = np.arange(indx_split, len(y))

    t_train = t[indx_train]
    y_train = y[indx_train]
    y_train = y_train.reshape(-1, 1)
    
    t_test = t[indx_test]
    y_test = y[indx_test]
    y_test = y_test.reshape(-1, 1)
    
    return t_train, y_train, t_test, y_test 

# %%
def windowed_dataset(y, input_window = 5, output_window = 1, stride = 1, num_features = 1):
  
    '''
    create a windowed dataset
    
    : param y:                time series feature (array)
    : param input_window:     number of y samples to give model 
    : param output_window:    number of future y samples to predict  
    : param stide:            spacing between windows   
    : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
    : return X, Y:            arrays with correct dimensions for LSTM
    :                         (i.e., [input/output window size # examples, # features])
    '''
  
    L = y.shape[0]
    num_samples = (L - input_window - output_window) // stride + 1

    X = np.zeros([input_window, num_samples, num_features])
    Y = np.zeros([output_window, num_samples, num_features])    
    
    for ff in np.arange(num_features):
        for ii in np.arange(num_samples):
            start_x = stride * ii
            end_x = start_x + input_window
            X[:, ii, ff] = y[start_x:end_x, ff]

            start_y = stride * ii + input_window
            end_y = start_y + output_window 
            Y[:, ii, ff] = y[start_y:end_y, ff]

    return X, Y

# %%
def numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest):
    '''
    convert numpy array to PyTorch tensor
    
    : param Xtrain:                           windowed training input data (input window size, # examples, # features); np.array
    : param Ytrain:                           windowed training target data (output window size, # examples, # features); np.array
    : param Xtest:                            windowed test input data (input window size, # examples, # features); np.array
    : param Ytest:                            windowed test target data (output window size, # examples, # features); np.array
    : return X_train_torch, Y_train_torch,
    :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors 

    '''
    
    X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
    Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

    X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
    Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)
    
    return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch

# %%
def SmoothMoves(file):
    #    if file[-3:] =='csv':
    ODataSet = np.genfromtxt(open(folder+file,'r'), delimiter=',',skip_header=0,missing_values=0,invalid_raise=False)
    SmoothX = denoise_wavelet(ODataSet[:,3], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
    #SmoothY = denoise_wavelet(ODataSet[:,4], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
    #SmoothZ = denoise_wavelet(ODataSet[:,5], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
    SmoothX -= np.average(SmoothX)
    Time = []
    for r in ODataSet:
        hr = int(r[1]/10000)
        mint = int((r[1]-10000*hr)/100)
        sec = r[1] % 100
        t = hr * 3600 + mint * 60 + sec + r[2]/10000.0
        Time.append(t)
    #SmoothY -= np.average(SmoothY)
    #SmoothZ -= np.average(SmoothZ)
    #MoveMatrix = np.matrix([SmoothX, SmoothY, SmoothZ])
    return SmoothX, np.asarray(Time)

# %%
#----------------------------------------------------------------------------------------------------------------
# window dataset

# set size of input/output windows 
iw = 90 
ow = 10 
s = 5

Xtrain = np.zeros(iw)
Ytrain = np.zeros(ow)
Xtest = np.zeros(iw)
Ytest = np.zeros(ow)

# %%
#----------------------------------------------------------------------------------------------------------------
# generate dataset for LSTM
files = ['230418 recording1.csv','230419 recording1.csv'] #,'230420 recording1.csv','230421 recording1.csv',
         #'230418 recording2.csv','230419 recording2.csv','230420 recording2.csv','230421 recording2.csv']

for file in files:
    y, t = SmoothMoves(files)

    t_train, y_train, t_test, y_test = train_test_split(t, y, split = 0.8)

    # generate windowed training/test datasets
    XTrain, YTrain= windowed_dataset(y_train, input_window = iw, output_window = ow, stride = s)
    XTest, YTest = windowed_dataset(y_test, input_window = iw, output_window = ow, stride = s)

    Xtrain = np.concatenate((Xtrain, XTrain), axis = 1)
    Ytrain = np.concatenate((Ytrain, YTrain), axis = 1)
    Xtest = np.concatenate((Xtest, XTest), axis = 1)
    Ytest = np.concatenate((Ytest, YTest), axis = 1)

# %%



# %%
#----------------------------------------------------------------------------------------------------------------
# LSTM encoder-decoder

# convert windowed data from np.array to PyTorch tensor
X_train, Y_train, X_test, Y_test = numpy_to_torch(Xtrain, Ytrain, Xtest, Ytest)

# %%
# specify model parameters and train
model = lstm_encoder_decoder.lstm_seq2seq(input_size = X_train.shape[2], hidden_size = 15)

# %%
loss = model.train_model(X_train, Y_train, n_epochs = 25, target_len = ow, batch_size = 1, training_prediction = 'mixed_teacher_forcing', teacher_forcing_ratio = 0.6, learning_rate = 0.01, dynamic_tf = False)

# %%
# plot predictions on train/test data
plotting.plot_train_test_results(model, Xtrain, Ytrain, Xtest, Ytest)

plt.close('all')

# %%
torch.save(model, "PyTorch_LSTM_xyz")



