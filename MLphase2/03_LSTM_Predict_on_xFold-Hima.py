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
import random

from time import time as ti

import CoreFunctions as cf
from skimage.restoration import denoise_wavelet

import os
import pickle

# currently running pid 4010813

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
    folder = '/sciclone/scr10/dchendrickson01/RecordingsSplit/xFold/'
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

def Openfile(file):
    try:
        ff = open(folder+file,'rb')
        dump = pickle.load(ff)
    
        return dump[0], dump[1]
    except:
        pass

location = folder
Titles = True
Ledgends = True

FileBatch = 500

num_cores = 30
num_gpus = 1

files = os.listdir(folder)
print('files: ', len(files))
random.shuffle(files)
print('files: ', len(files))


st = ti()


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
    
  def build_model(self):
    timesteps = self.timesteps
    n_features = self.n_features
    model = Sequential()
    
    # Padding
    #model.add(Masking(mask_value=0.0, input_shape=(timesteps, n_features)))

    # Encoder
    model.add(LSTM(timesteps, activation='relu', input_shape=(TimeSteps, Features), return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(LSTM(12, activation='relu'))
    model.add(RepeatVector(timesteps))
    
    # Decoder
    model.add(LSTM(timesteps, activation='relu', return_sequences=True))
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
    model.summary()
    self.model = model
    
  def simple_model(self):
    
    # define model
    model = Sequential()
    model.add(LSTM(500, input_shape=(TimeSteps, Features), return_sequences=True))
    #model.add(RepeatVector(TimeSteps))
    #model.add(RepeatVector(PredictSize))
    
    #model.add(LSTM(25, return_sequences=True))
    
    model.add(Lambda(lambda x: x[:, -PredictSize:, :])) #Select last N from output  
    #https://stackoverflow.com/questions/43034960/many-to-one-and-many-to-many-lstm-examples-in-keras?noredirect=1&lq=1
    
    model.add(TimeDistributed(Dense( self.n_features, activation='softmax')))
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


lstm_autoencoder2 = LSTM_Autoencoder(optimizer='adam', loss='mse')

from tensorflow.python.keras import backend as K
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
                        inter_op_parallelism_threads=num_cores, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_cores,
                                        'GPU' : num_gpus}
                       )
session = tf.compat.v1.Session(config=config)
K.set_session(session)

lstm_autoencoder2.simple_model()


Loops=int(len(files)/FileBatch)+1

for k in range(Loops):

    print("Starting Loop "+str(k+1)+" of "+str(Loops+1))
    
    Moves = []
    #Names = []
    
    start = k * FileBatch
    
    Results = Parallel(n_jobs=num_cores)(delayed(Openfile)(file) for file in files)

    print('Results in from Parallel', len(Results))

    i=0

    for result in Results:
        for j in range(len(result[0])):
            Moves.append(result[0][j,:,:])
            #Names.append(result[1]+str(i).zfill(5))
            i+=1
    del Results
    
    print('Results parsed ', print(len(Names)))

    TimeSteps = 700
    PredictSize = 50
    Features = 3

    X, y = list(), list()
    for move in Moves:
        X.append(move[:TimeSteps,:])
        y.append(move[TimeSteps:,:])
    print(np.shape(y))
    
    del Moves
    
    Batches = 64
    #NumbBatches = 100

    lstm_autoencoder2.model.fit(X, y, epochs=4, batch_size=Batches, verbose=1)
    lstm_autoencoder2.model.save("LSTM_AtOnce_700p50"+str(k).zfill(3))

'''
st = ti()
SamplesPerSet = Batches * NumbBatches

SetsNeeded = int(len(X) / SamplesPerSet)
if  int(len(X) / SamplesPerSet) != 0:
    SetsNeeded += 1
print(len(X), SetsNeeded)

PercentPerSet = 1.0 / float(SetsNeeded)

PercentHoldOutForNext=1.0
for i in range(SetsNeeded-1):
    PercentHoldOutForNext = 1.0 - (SamplesPerSet / len(X))
    seq_train, seq_test, out_train, out_test = train_test_split(X, y, test_size=PercentHoldOutForNext, shuffle=True, random_state=0)
    seq_train = np.asarray(seq_train)
    out_train = np.asarray(out_train)
    if i == 0:
        vb = 1
    else:
        vb = 0
    lstm_autoencoder2.model.fit(seq_train, out_train, epochs=2, batch_size=Batches, verbose=vb)
    MoveSegments = seq_test
    NextDataPoint = out_test
    print(str(i+1)+' of ' + str(SetsNeeded), (int(ti()-st)/6)/10, (int(((ti()-st)/(i+1) * ( SetsNeeded -1) - (ti()-st) )/6)/10))


lstm_autoencoder2.model.save("LSTM_predict_full_700p50")
'''