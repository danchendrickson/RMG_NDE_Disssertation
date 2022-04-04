
import numpy as np
import matplotlib.pyplot as plt

import os
import random
import datetime

import multiprocessing
from joblib import Parallel, delayed


import keras
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
 
import cv2
from sklearn.model_selection import train_test_split
import keras_metrics as km
  
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import multilabel_confusion_matrix

import tensorflow as tf

SensorPositonFile = 'D:\\SensorLocations.csv'
folder = 'D:\\CraneData\\'

img_height , img_width = 3, 100
FrameLength = img_width
NumberFrames = 600

OutputVectors = np.genfromtxt(open(SensorPositonFile,'r'), delimiter=',',skip_header=1,missing_values=0)

def truthVector(Filename):
    # Parses the filename, and compares it against the record of sensor position on cranes
    # inputs: filename
    # outputs: truth vector


    #Parsing the file name.  Assuming it is in the standard format
    sYear = '20' + Filename[10:12]
    sMonth = Filename[12:14]
    sDay = Filename[14:16]
    sHour = Filename[17:19]
    sMin = Filename[19:21]
    sSensor = Filename[23]

    mask = []

    i=0
    #loops through the known sensor movements, and creates a filter mask
    for spf in OutputVectors:
        
        startDate = datetime.date(int(spf[0]), int(spf[1]), int(spf[2])) + datetime.timedelta(hours=spf[3]) + datetime.timedelta(minutes=spf[4])
        endDate = datetime.date(int(spf[5]), int(spf[6]), int(spf[7])) + datetime.timedelta(hours=spf[8]) + datetime.timedelta(minutes=spf[9])
        sDate = datetime.date(int(sYear), int(sMonth), int(sDay)) + datetime.timedelta(hours=int(sHour)) + datetime.timedelta(minutes=int(sMin))

        if sDate >= startDate and sDate <= endDate and int(spf[10]) == int(sSensor):
            mask.append(True)
            i+=1
        else:
            mask.append(False)
        
    if i == 0: print('error ', Filename)

    results = OutputVectors[mask,11:]

    if i > 1: 
        print('Found Two ', Filename)
        results = results[0,:]
    #np.array(results)

    return results

def makeFrames(input,sequ,frameLength):
    frames=[] #np.array([],dtype=object,)
    segmentGap = int((np.shape(input)[0]-frameLength)/sequ)
    #print(segmentGap,sequ, frameLength)
    for i in range(sequ):
        start = i * segmentGap
        imageMatrix = input[start:start+frameLength,:]
        np.matrix(imageMatrix)
        imageMatrix = imageMatrix.T
        frames.append(imageMatrix)
    
    return frames


def MakeData(Filename):

    fileData = np.genfromtxt(open(folder+Filename,'r'), delimiter=',',skip_header=0,missing_values=0).T[2:5,:]
    
    for i in range(3):
        norm = np.average(fileData[i,:])
        fileData[i,:]-=norm

    frames = makeFrames(fileData, NumberFrames,FrameLength)
    frames = np.asarray(frames)
    
    results = truthVector(Filename)

    return frames, results

n = 25

files = os.listdir(folder)
files = random.sample(files,n)

#DataSet = [] #np.array([],dtype=object,)

#ResultsSet = np.zeros((len(files),np.shape(OutputVectors[:,11:])[1])) #np.array([],dtype=object,)

i=0

# for filename in files:
#     if filename[-3:] == 'csv':
#         ResultsSet[i,:] = truthVector(filename)
#         fileData = np.genfromtxt(open(folder+filename,'r'), delimiter=',',skip_header=0,missing_values=0).T[2:5,:]
#         frames = makeFrames(fileData.T,numberFrames,img_width)
#         #print(np.shape(DataSet),np.shape(frames),1)
#         frames = np.asarray(frames)
#         DataSet.append(frames)
#         i+=1
#     else: print(filename[-3:])

num_cores = 4 # multiprocessing.cpu_count() - 1
Data = Parallel(n_jobs=num_cores)(delayed(MakeData)(file) for file in files)

DataSet, ResultsSet = Data[0,:], Data[1,:]

#ResultsSet = np.asarray(ResultsSet)
DataSet = np.asarray(DataSet)

ResultSubSet = [6,7,8,11]

ResultsSet = ResultsSet[0:np.shape(DataSet)[0],ResultSubSet]

print('Data Made')

X_train, X_test, y_train, y_test = train_test_split(DataSet, ResultsSet, test_size=0.20, shuffle=True, random_state=0)

model = Sequential()
model.add(ConvLSTM2D(filters = 64, 
            kernel_size = (3, 3), 
            return_sequences = False, 
            data_format = "channels_last", 
            input_shape = (NumberFrames, img_height, img_width, 1)
            )
        )
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(128, actication="reul"))
model.add(Dense(np.shape(y_train)[1], activation = "softmax"))
 
model.summary()
 
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
 
earlystop = EarlyStopping(patience=7)   
callbacks = [earlystop]

 
history = model.fit(x = X_train, y = y_train, epochs=40, batch_size = 8 , shuffle=False, validation_split=0.2, callbacks=callbacks)

