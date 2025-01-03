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
#from skimage.restoration import denoise_wavelet

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
    folder = '/scratch/RecordingsSplit/xFold/'
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
        print("bad file ",file)


location = folder
Titles = True
Ledgends = True

FileBatch = 20000

TimeSteps = 750
PredictSize = 25
Features = 3

MiddleLayerSize = 50
CompressedVectorSize = 10

Batches = 32

num_cores = 30
num_gpus = 0


from sys import getsizeof

from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Masking, Lambda, Input
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
    model.add(Input(shape=(TimeSteps, Features)))
    model.add(LSTM(timesteps, activation='relu', return_sequences=True))
    model.add(LSTM(MiddleLayerSize, activation='relu', return_sequences=True))
    model.add(LSTM(CompressedVectorSize, activation='relu'))
    model.add(RepeatVector(timesteps))
    
    # Decoder
    model.add(LSTM(timesteps, activation='relu', return_sequences=True))
    model.add(LSTM(MiddleLayerSize, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    
    model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
    model.summary()
    self.model = model
    
  def fit(self, X, epochs=3, batch_size=32):
    #self.timesteps = np.shape(X)[0]
    #self.build_model()
    
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

from tensorflow.keras.utils import Sequence

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        super().__init__(workers = num_cores, use_multiprocessing=True, max_queue_size=1)
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y

lstm_autoencoder2.build_model()


files = os.listdir(folder)
print('files: ', len(files))

TotalMoves=0
Loops=int(len(files)/FileBatch)+1

k=0
start = k * FileBatch
random.shuffle(files)

files.remove("230208 recording4-000234.p")

Results = Parallel(n_jobs=num_cores)(delayed(Openfile)(file) for file in files[start:start+FileBatch])

X = list()

for result in Results:
    try:
        for j in range(len(result[0])):
            X.append(result[0][j,:,:])
            TotalMoves+=1
    except:
        print(np.shape(result[0]),result[1])
del Results

getsizeof(X)

X = tf.convert_to_tensor(X, np.float32)

getsizeof(X)

lstm_autoencoder2.model.fit(X,X, epochs=4, batch_size=Batches, verbose=2)
lstm_autoencoder2.model.save("LSTM_750_self.keras")

print('Total Moves ',TotalMoves)

tic = ti()

for j in range(15):

    random.shuffle(files)

    for k in range(Loops):
        print("Starting Loop "+str(k+1)+" of "+str(Loops+1)+" elapsted time: "+str(int((ti()-tic)/100)*100))

        start = k * FileBatch

        Results = Parallel(n_jobs=num_cores)(delayed(Openfile)(file) for file in files[start:start+FileBatch])

        X = list()
        
        for result in Results:
            #try:
            for j in range(len(result[0])):
                X.append(result[0][j,:,:])
                TotalMoves+=1
            #except:
            #    print(np.shape(result[0]),result[1])
        del Results



        with tf.device('/cpu:0'):
            X = tf.convert_to_tensor(X, np.float32)
            #y = tf.convert_to_tensor(y, np.float32)

        test_gen = DataGenerator(X, X, Batches)
    
            
        lstm_autoencoder2.model.fit(test_gen, epochs=4, batch_size=Batches, verbose=2)
        lstm_autoencoder2.model.save("LSTM_750_self.keras")

        print('Total Moves ',TotalMoves)

        del X