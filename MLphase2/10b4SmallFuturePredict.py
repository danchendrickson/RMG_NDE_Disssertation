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

import CoreFunctions as cf
from skimage.restoration import denoise_wavelet

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

Saving = False
location = folder
Titles = True
Ledgends = True

f = 0


files = ['230418 recording1.csv','230419 recording1.csv','230420 recording1.csv','230421 recording1.csv',
         '230418 recording2.csv','230419 recording2.csv','230420 recording2.csv','230421 recording2.csv']

#Smooth = cf.Smoothing(ODataSet[:,3],2) #,50)
def SmoothMoves(file):
    #    if file[-3:] =='csv':
    ODataSet = np.genfromtxt(open(folder+file,'r'), delimiter=',',skip_header=0,missing_values=0,invalid_raise=False)
    SmoothX = denoise_wavelet(ODataSet[:,3], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
    SmoothY = denoise_wavelet(ODataSet[:,4], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
    SmoothZ = denoise_wavelet(ODataSet[:,5], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
    SmoothX -= np.average(SmoothX)
    SmoothY -= np.average(SmoothY)
    SmoothZ -= np.average(SmoothZ)
    MoveMatrix = np.matrix([SmoothX, SmoothY, SmoothZ])
    return MoveMatrix
    #else:
    #    pass


LoopFiles = 8
loops = int(len(files) / LoopFiles) 
if len(files)%LoopFiles != 0:
    loops += 1



st = ti()

Moves = []

for k in range(loops):
    if k == loops -1:
        tfiles = files[k*LoopFiles:]
    else:
        tfiles = files[k*LoopFiles:(k+1)*LoopFiles]
    #Results = Parallel(n_jobs=LoopFiles)(delayed(DeviationVelocity)(file) for file in tfiles)
    Results = Parallel(n_jobs=LoopFiles)(delayed(SmoothMoves)(file) for file in tfiles)
    #Results =[]
    #for file in tfiles:
    #    Results.append(DeviationVelocity(file))
    #    print(file, (ti()-st)/60.0)
    for result in Results:
        Moves.append(result)
    print(k, (ti()-st)/60.0)
        
TimeSteps = 1000
StepSize = 12
PredictSize = 25
Features = 3
Features = np.shape(Moves[0])[0]

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps, s_step = 1, y_steps = 1):
    X, y = list(), list()
    Steps_to_take = int(len(sequences) / s_step)
    for j in range(Steps_to_take):
        i = j * s_step
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x = sequences[i:end_ix, :]
        seq_y = sequences[end_ix:end_ix+y_steps :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

Sequences = []
Outputs = []
for move in Moves:
    Seq, Out = split_sequences(move.T,TimeSteps,StepSize,PredictSize)
    Sequences.append(Seq)
    Outputs.append(Out)
    

MoveSegments = []
for seq in Sequences:
    for mv in seq:
        MoveSegments.append(mv)
NextDataPoint = []
for out in Outputs:
    for pt in out:
        NextDataPoint.append(pt) #np.reshape(pt,(PredictSize,3)))

print('Move Segments ', len(MoveSegments),(ti()-st)/60.0)

from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Masking, Lambda
from keras.models import Sequential
import tensorflow as tf

class LSTM_Autoencoder:
  def __init__(self, optimizer='adam', loss='mse'):
    self.optimizer = optimizer
    self.loss = loss
    self.n_features = Features
    self.timesteps = TimeSteps
    
  '''def build_model(self):
    timesteps = self.timesteps
    n_features = self.n_features
    model = Sequential()
    
    # Padding
    #model.add(Masking(mask_value=0.0, input_shape=(timesteps, n_features)))

    # Encoder
    model.add(LSTM(timesteps, activation='relu', input_shape=(TimeSteps, Features), return_sequences=True))
    model.add(LSTM(35, activation='relu', return_sequences=True))
    model.add(LSTM(6, activation='relu'))
    model.add(RepeatVector(timesteps))
    
    # Decoder
    model.add(LSTM(timesteps, activation='relu', return_sequences=True))
    model.add(LSTM(35, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
    model.summary()
    self.model = model'''
    
  def simple_model(self):
    
    # define model
    model = Sequential()
    model.add(LSTM(362, input_shape=(TimeSteps, Features), return_sequences=True))
    model.add(LSTM(3, input_shape=(TimeSteps, Features), return_sequences=True))
    #model.add(RepeatVector(TimeSteps))
    #model.add(RepeatVector(PredictSize))
    
    #model.add(LSTM(25, return_sequences=True))
    
    model.add(Lambda(lambda x: x[:, -PredictSize:, :])) #Select last N from output  
    #https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras?noredirect=1&lq=1
    
    #model.add(TimeDistributed(Dense( self.n_features, activation='softmax')))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    self.model = model
    
  def fit(self, X, epochs=3, batch_size=32):
    #self.timesteps = np.shape(X)[0]
    self.build_model()
    
    #input_X = np.expand_dims(X, axis=1)
    self.model.fit(X, X, epochs=epochs, batch_size=batch_size)
    
  def predict(self, X):
    #input_X = np.expand_dims(X, axis=1)
    output_X = self.model.predict(X)
    reconstruction = np.squeeze(output_X)
    return np.linalg.norm(X - reconstruction, axis=-1)
  
  def plot(self, scores, timeseries, threshold=0.95):
    sorted_scores = sorted(scores)
    threshold_score = sorted_scores[round(len(scores) * threshold)]
    
    plt.title("Reconstruction Error")
    plt.plot(scores)
    plt.plot([threshold_score]*len(scores), c='r')
    plt.show()
    
    anomalous = np.where(scores > threshold_score)
    normal = np.where(scores <= threshold_score)
    
    plt.title("Anomalies")
    plt.scatter(normal, timeseries[normal][:,-1], s=3)
    plt.scatter(anomalous, timeseries[anomalous][:,-1], s=5, c='r')
    plt.show()

lstm_autoencoder2 = LSTM_Autoencoder(optimizer='adam', loss='mean_absolute_error')
lstm_autoencoder2.simple_model()

Batches = 32
NumbBatches = 1000

SamplesPerSet = Batches * NumbBatches

SetsNeeded = int(len(MoveSegments) / SamplesPerSet)
if  int(len(MoveSegments) / SamplesPerSet) != 0:
    SetsNeeded += 1
print(len(MoveSegments), SetsNeeded)

PercentPerSet = 1.0 / float(SetsNeeded)

PercentHoldOutForNext=1.0

st = ti()

for i in range(SetsNeeded-1):
    PercentHoldOutForNext = 1.0 - (SamplesPerSet / len(MoveSegments))
    seq_train, seq_test, out_train, out_test = train_test_split(MoveSegments, NextDataPoint, test_size=PercentHoldOutForNext, shuffle=True, random_state=0)
    seq_train = np.asarray(seq_train)
    out_train = np.asarray(out_train)
    if i == 0 or i == SetsNeeded-2:
        vb = 1
    else:
        vb = 0
    try:
        lstm_autoencoder2.model.fit(seq_train, out_train, epochs=2, batch_size=1, verbose=vb)
    except:
        print('Something went wrong at this loop')
    MoveSegments = seq_test
    NextDataPoint = out_test
    print(str(i+1)+' of ' + str(SetsNeeded), (ti()-st)/60, (((ti()-st)/(i+1) * ( SetsNeeded -1) - (ti()-st) )/60/60))


lstm_autoencoder2.model.save("LSTMmaxParam1BatchMaeLlongpredict")
