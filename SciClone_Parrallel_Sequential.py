
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

rootFolder = "D:\\"

SensorPositonFile = rootFolder + 'SensorStatsSmall.csv'
folder = rootFolder + 'CraneData\\'
SaveModelFolder = rootFolder + 'SavedModel\\'

img_height , img_width = 3, 100
FrameLength = img_width
numberFrames = 1200
NumberOfFiles = 250
num_cores = multiprocessing.cpu_count() -1

OutputVectors = np.genfromtxt(open(SensorPositonFile,'r'), delimiter=',',skip_header=1,dtype=int, missing_values=0)

def truthVector(Filename):
    # Parses the filename, and compares it against the record of sensor position on cranes
    # inputs: filename
    # outputs: truth vector


    #Parsing the file name.  Assuming it is in the standard format
    sSensor = Filename[23]
    sDate = datetime.datetime.strptime('20'+Filename[10:21],"%Y%m%d-%H%M")

    mask = []

    i=0
    #loops through the known sensor movements, and creates a filter mask
    for spf in OutputVectors:
        
        startDate = datetime.datetime.strptime(str(spf[0])+str(spf[1]).zfill(2)+str(spf[2]).zfill(2)
            +str(spf[3]).zfill(2)+str(spf[4]).zfill(2),"%Y%m%d%H%M")
        #datetime.date(int(spf[0]), int(spf[1]), int(spf[2])) + datetime.timedelta(hours=spf[3]) + datetime.timedelta(minutes=spf[4])
        endDate = datetime.datetime.strptime(str(spf[5])+str(spf[6]).zfill(2)+str(spf[7]).zfill(2)
            +str(spf[8]).zfill(2)+str(spf[9]).zfill(2),"%Y%m%d%H%M")
        #datetime.date(int(spf[5]), int(spf[6]), int(spf[7])) + datetime.timedelta(hours=spf[8]) + datetime.timedelta(minutes=spf[9])
        
        if sDate >= startDate and sDate <= endDate and int(spf[10]) == int(sSensor):
            mask.append(True)
            i+=1
        else:
            mask.append(False)
        
    if i != 1: print('error ', i, Filename)

    results = OutputVectors[mask,11:]

    if i > 1: 
        print('Found Two ', Filename)
        results = results[0,:]
    #np.array(results)

    return results

def makeFrames(input): #,sequ,frameLength):
    frames=[] #np.array([],dtype=object,)
    segmentGap = int((np.shape(input)[0]-FrameLength)/numberFrames)
    #print(segmentGap,sequ, frameLength)
    for i in range(numberFrames):
        start = i * segmentGap
        imageMatrix = input[start:start+FrameLength,:]
        np.matrix(imageMatrix)
        imageMatrix = imageMatrix.T
        frames.append(imageMatrix)
    
    return frames

def ParseFile(Filename):

    Results = truthVector(Filename)
    
    fileData = np.genfromtxt(open(folder+Filename,'r'), delimiter=',',skip_header=0,missing_values=0).T[2:5,:]
    frames = makeFrames(fileData.T) #,numberFrames,img_width)
    frames = np.asarray(frames)
    
    return frames, Results

files = os.listdir(folder)
files = random.sample(files,NumberOfFiles)

print('Sample Created')

DataSet = [] 

ResultsSet = np.zeros((len(files),np.shape(OutputVectors[:,11:])[1]))

i=0

Data = Parallel(n_jobs=num_cores)(delayed(ParseFile)(file) for file in files)

DataSet = [] 
i=0
for datum in Data:
    DataSet.append(datum[0])
    ResultsSet[i]=datum[1][0]
    i+=1

DataSet = np.asarray(DataSet)

print('Data Parsed')

#ResultsSet = ResultsSet[0:np.shape(DataSet)[0],:]

X_train, X_test, y_train, y_test = train_test_split(DataSet, ResultsSet, test_size=0.20, shuffle=True, random_state=0)

model = Sequential()
model.add(ConvLSTM2D(filters = 64, 
            kernel_size = (3, 3), 
            return_sequences = False, 
            data_format = "channels_last", 
            input_shape = (numberFrames, img_height, img_width, 1)
            )
        )
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(np.shape(y_train)[1], activation = "softmax"))
 
model.summary()
 
opt = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=["accuracy"])
 
earlystop = EarlyStopping(patience=7)   
callbacks = [earlystop]

history = model.fit(x = X_train, y = y_train, epochs=40, batch_size = 8 , shuffle=False, validation_split=0.2, callbacks=callbacks)

model.save(SaveModelFolder)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(rootFolder + 'ModelAccuracy.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(rootFolder + 'ModelLoss.png')
plt.show()

