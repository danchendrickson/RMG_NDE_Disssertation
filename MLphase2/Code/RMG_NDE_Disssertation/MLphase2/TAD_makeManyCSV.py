import numpy as np
import pandas as pd
#import keras
#from keras import layers
#from matplotlib import pyplot as plt
#import numpy as np
import matplotlib.pyplot as plt
#from cycler import cycler
import scipy.special as sp
import os as os
#import pywt as py
#import statistics as st
import os as os
#import random
import multiprocessing
from joblib import Parallel, delayed
#import platform
from time import time as ti
from skimage.restoration import denoise_wavelet
#import tensorflow as tf
import pickle
import CoreFunctions as cf

import sys

DataFolder = '/sciclone/scr10/dchendrickson01/Recordings2/'
SaveFolder = '/sciclone/scr10/dchendrickson01/750inputs/'

StartPoint = int(sys.argv[1])
Groups = 180

verbose = sys.argv[2]
small = sys.argv[3]
noise = verbose

TIME_STEPS = 750
Skips = 5

tic = ti()
start = tic

def RollingStdDev(RawData, SmoothData, RollSize = 25):
    StdDevs = []
    for i in range(RollSize):
        Diffs = RawData[0:i+1]-SmoothData[0:i+1]
        Sqs = Diffs * Diffs
        Var = sum(Sqs) / (i+1)
        StdDev = np.sqrt(Var)
        StdDevs.append(StdDev)
    for i in range(len(RawData)-RollSize-1):
        j = i + RollSize
        Diffs = RawData[i:j]-SmoothData[i:j]
        Sqs = Diffs * Diffs
        Var = sum(Sqs) / RollSize
        StdDev = np.sqrt(Var)
        StdDevs.append(StdDev)  
    
    return StdDevs

def RollingStdDevFaster(RawData, SmoothData, RollSize = 25):

    Diffs = RawData - SmoothData
    
    del RawData, SmoothData
    
    Sqs = Diffs * Diffs

    del Diffs
    
    Sqs = Sqs.tolist() 
    
    Sqs.extend(np.zeros(RollSize))
    
    mSqs = np.matrix(Sqs)
    
    for i in range(RollSize):
        Sqs.insert(0, Sqs.pop())
        mSqs = np.concatenate((np.matrix(Sqs),mSqs))
    
    sVect = mSqs.sum(axis=0)
    eVect = (mSqs!=0).sum(axis=0)
    
    del mSqs, Sqs
    
    VarVect = sVect / eVect
    
    StdDevs = np.sqrt(VarVect)
    
    return StdDevs[:-RollSize]


def RollingSum(Data, Length = 100):
    RollSumStdDev = []
    for i in range(Length):
        RollSumStdDev.append(sum(Data[0:i+1]))
    for i in range(len(Data) - Length):
        RollSumStdDev.append(sum(Data[i:i+Length]))
    return RollSumStdDev

def SquelchPattern(DataSet, StallRange = 5000, SquelchLevel = 0.02, verbose = False):
    
    SquelchSignal = np.ones(len(DataSet))
    if verbose:
        print(len(SquelchSignal))
        
    for i in range(len(DataSet)-2*StallRange):
        if np.average(DataSet[i:i+StallRange]) < SquelchLevel:
            SquelchSignal[i+StallRange]=0

    return SquelchSignal

def SquelchPatternFast(DataSet, StallRange = 5000, SquelchLevel = 0.02):
    SquelchSignal = np.ones(len(DataSet))

    #DataSet = DataSet.tolist() 
    
    DataSet.extend(np.zeros(StallRange))
    
    DSM = np.matrix(DataSet)
    
    for i in range(StallRange):
        DataSet.insert(0, DataSet.pop())
        DSM = np.concatenate((np.matrix(DataSet),DSM))
    
    DsmAvs = np.average(DSM,axis=0)
    
    DsmAvs[DsmAvs < SquelchLevel] = 0
    DsmAvs[DsmAvs >= SquelchLevel] = 1

    return SquelchSignal

def getVelocity(Acceleration, Timestamps = 0.003, Squelch = [], corrected = 0):
    velocity = np.zeros(len(Acceleration))
    
    Acceleration -= np.average(Acceleration)
    
    if len(Timestamps) == 1:
        dTime = np.ones(len(Acceleration),dtype=float) * Timestamps
    elif len(Timestamps) == len(Acceleration):
        dTime = np.zeros(len(Timestamps), dtype=float)
        dTime[0]=1
        for i in range(len(Timestamps)-1):
            j = i+1
            if float(Timestamps[j]) > float(Timestamps[i]):
                dTime[j]=float(Timestamps[j])-float(Timestamps[i])
            else:
                dTime[j]=float(Timestamps[j])-float(Timestamps[i])+10000.0
        dTime /= 10000.0

    velocity[0] = Acceleration[0] * (dTime[0])

    for i in range(len(Acceleration)-1):
        j = i + 1
        if corrected ==2:
            if Squelch[j]==0:
                velocity[j]=0
            else:
                velocity[j] = velocity[i] + Acceleration[j] * dTime[j]                
        else:
            velocity[j] = velocity[i] + Acceleration[j] * dTime[j]

    if corrected == 1:
        PointVairance = velocity[-1:] / len(velocity)
        for i in range(len(velocity)):
            velocity[i] -=  PointVairance * i
    
    velocity *= 9.81

    return velocity

def MakeDTs(Seconds, Miliseconds):
    dts = np.zeros(len(Miliseconds), dtype=float)
    dts[0]=1
    for i in range(len(MiliSeconds)-1):
        j = i+1
        if Seconds[j]==Seconds[i]:
            dts[j]=Miliseconds[j]-Miliseconds[i]
        else:
            dts[j]=Miliseconds[j]-Miliseconds[i]+1000
    dts /= 10000
    return dts


def split_list_by_ones(original_list, ones_list):
    # Created with Bing AI support
    #  1st request: "python split list into chunks based on value"
    #  2nd request: "I want to split the list based on the values in a second list.  Second list is all 1s and 0s.  I want all 0s removed, and each set of consequtive ones as its own item"
    #  3rd request: "That is close.  Here is an example of the two lists, and what I would want returned: original_list = [1, 2, 3, 8, 7, 4, 5, 6, 4, 7, 8, 9]
    #                ones_list =     [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    #                return: [[1, 2, 3, 8], [4, 5, 6], [8,9]]"
    #
    #This is the function that was created and seems to work on the short lists, goin to use fo rlong lists
    
    result_sublists = []
    sublist = []

    for val, is_one in zip(original_list, ones_list):
        if is_one:
            sublist.append(val)
        elif sublist:
            result_sublists.append(sublist)
            sublist = []

    # Add the last sublist (if any)
    if sublist:
        result_sublists.append(sublist)

    return result_sublists

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS, skips = Skips):
    output = []
    for i in range(int((len(values) - time_steps + skips)/skips)):
        output.append(values[i*skips : (i*skips + time_steps)])
    return np.stack(output)

def runFile(file, verbose = False, small = False, index=0, start=ti()):
    noise = verbose
    if file[-4:] == '.csv':    
        try:
            dataset = pd.read_csv(DataFolder+file, delimiter =", ", header=None, engine='python',on_bad_lines='skip')
            if noise:
                print("File Read", ti()-start)
            dataset = dataset.rename(columns={0:"Day"})
            dataset = dataset.rename(columns={1:"Second"})
            dataset = dataset.rename(columns={2:"FracSec"})
            dataset = dataset.rename(columns={3:"p"})
            dataset = dataset.rename(columns={4:"h"})
            dataset = dataset.rename(columns={5:"v"})
            dataset = dataset.rename(columns={6:"Sensor"})

            dataset['Second'].replace('',0)
            dataset[['Day','Second']] = dataset[['Day','Second']].apply(lambda x: x.astype(int).astype(str).str.zfill(6))
            dataset[['FracSec']] = dataset[['FracSec']].apply(lambda x: x.astype(int).astype(str).str.zfill(4))

            dataset["timestamp"] = pd.to_datetime(dataset.Day+dataset.Second+dataset.FracSec,format='%y%m%d%H%M%S%f')
            dataset["timestamps"] = dataset["timestamp"]

            dataset["p"] = dataset.p - np.average(dataset.p)
            dataset["h"] = dataset.h - np.average(dataset.h)
            dataset["v"] = dataset.v - np.average(dataset.v)
            #dataset["r"] = np.sqrt(dataset.p**2 + dataset.h**2 + dataset.v**2)

            dataset.index = dataset.timestamp

            dataset["SmoothP"] = denoise_wavelet(dataset.p, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
            dataset["SmoothH"] = denoise_wavelet(dataset.h, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
            dataset["SmoothV"] = denoise_wavelet(dataset.v, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')

            if noise:
                print("Data Cleaned", ti()-start, len(dataset.p))

            RawData = dataset.v
            SmoothData = dataset.SmoothV
            RollSize = 25

            Diffs = RawData - SmoothData

            Sqs = Diffs * Diffs

            Sqs = Sqs.tolist() 

            Sqs.extend(np.zeros(RollSize))

            mSqs = np.matrix(Sqs)

            for i in range(RollSize):
                Sqs.insert(0, Sqs.pop())
                mSqs = np.concatenate((np.matrix(Sqs),mSqs))

            sVect = mSqs.sum(axis=0)
            eVect = (mSqs!=0).sum(axis=0)

            VarVect = sVect / eVect

            StdDevs = np.sqrt(VarVect)

            StdDevsZ = np.asarray(StdDevs)

            StdDevsZ=np.append(StdDevsZ,[0])

            StdDevsZ = np.asarray(StdDevsZ.T[:len(dataset.p)])

            if noise:
                print("Size StdDevsZ", ti()-start, np.shape(StdDevsZ))

            #StdDevsZ = np.nan_to_num(StdDevsZ)

            #StdDevsZ[StdDevsZ == np.inf] = 0
            #StdDevsZ[StdDevsZ == -np.inf] = 0

            if noise:
                print("cleaned", ti()-start, np.shape(StdDevsZ))

            SmoothDevZ = denoise_wavelet(StdDevsZ, method='VisuShrink', mode='soft', wavelet='sym2', rescale_sigma='True')

            if noise:
                print("denoise 1", ti()-start, np.shape(StdDevsZ))

            #SmoothDevZa = cf.Smoothing(StdDevsZ, 3, wvt='sym2', dets_to_remove=2, levels=3)
            #SmoothDevZ = np.ravel(SmoothDevZ[0,:])

            #SmoothDevZ = SmoothDevZ.tolist()

            if noise:
                print("denoise 2", ti()-start, np.shape(SmoothDevZ))

            #ataset["SmoothDevZ"] = SmoothDevZ


            Max = np.max(SmoothDevZ)
            if noise:
                print("Max", ti()-start, np.shape(Max), Max)

            buckets = int(Max / 0.005) + 1
            bins = np.linspace(0,buckets*0.005,buckets+1)
            counts, bins = np.histogram(SmoothDevZ,bins=bins)

            CummCount = 0
            HalfWay = 0
            for i in range(len(counts)):
                CummCount += counts[i]
                if CummCount / len(SmoothDevZ) >= 0.5:
                    if HalfWay == 0:
                        HalfWay = i

            SquelchLevel = bins[HalfWay] 
            if noise:
                print("SmoothDevz size", np.shape(SmoothDevZ))

            dataset["IsMoving"] = SquelchPattern(SmoothDevZ, 4000, SquelchLevel, verbose=noise)

            if noise:
                print("Squelch Made", ti()-start)
            #dataset["velocity"] = getVelocity(dataset.p, dataset.FracSec, dataset.IsMoving, 2)
            #if noise:
            #    print("Velocity Calculated.  File done: ",file)

            df_pr = split_list_by_ones(dataset.p, dataset.IsMoving)
            df_hr = split_list_by_ones(dataset.h, dataset.IsMoving)
            df_vr = split_list_by_ones(dataset.v, dataset.IsMoving)
            df_ps = split_list_by_ones(dataset.SmoothP, dataset.IsMoving)
            df_hs = split_list_by_ones(dataset.SmoothH, dataset.IsMoving)
            df_vs = split_list_by_ones(dataset.SmoothV, dataset.IsMoving)

            if verbose:
                print("Split by ones", ti()-start)


            df_p=[0]
            df_h=[0]
            df_v=[0]
            df_rp=[0]
            df_rh=[0]
            df_rv=[0]
            for i in range(len(df_ps)):
                df_p += df_ps[i]
                df_h += df_hs[i]
                df_v += df_vs[i]
                df_rp += df_pr[i]
                df_rh += df_hr[i]
                df_rv += df_vr[i]

            if verbose:
                print('format changed', ti()-start, np.shape(df_p))

            del df_pr, df_hr, df_vr, df_hs, df_ps, df_vs
            
            training_mean = np.average(df_p)
            training_std = np.std(df_p)
            df_training_value_p = (df_p - training_mean) 
            df_training_value_p = df_training_value_p / training_std

            if verbose:
                print('p', ti()-start)

            training_mean = np.average(df_h)
            training_std = np.std(df_h)
            df_training_value_h = (df_h - training_mean) 
            df_training_value_h /= training_std

            if verbose:
                print('h', ti()-start)

            training_mean = np.average(df_v)
            training_std = np.std(df_v)
            df_training_value_v = (df_v - training_mean) 
            df_training_value_v /= training_std

            del df_p, df_h, df_v
            
            if verbose:
                print('Data normalized', ti()-start)


            x_train_p = create_sequences(df_rp)
            x_train_h = create_sequences(df_rh)
            x_train_v = create_sequences(df_rv)

            x_train_ps = create_sequences(df_training_value_p)
            x_train_hs = create_sequences(df_training_value_h)
            x_train_vs = create_sequences(df_training_value_v)


            if verbose:
                print('Sequences created', ti()-start)

            if small:
                dataLength = 100
            else:
                dataLength=len(x_train_ps)

            for i in range(dataLength):
                temp = np.matrix([x_train_p[i],x_train_h[i],x_train_v[i],x_train_ps[i],x_train_hs[i],x_train_vs[i]]) #
                np.savetxt(SaveFolder+file[-4:]+str(i).zfill(6)+'.csv', temp, delimiter=",")

            del x_train_ps, x_train_hs, x_train_vs


            print('Done, files saved',index, file, ti()-start)
        except:
            print(file, "Is bad", index)


files = os.listdir(DataFolder)

print("Starting")

#Results = Parallel(n_jobs=8)(delayed(runFile)(file, False, False, index) for index, file in enumerate(files))

for index in range(Groups):
    start = StartPoint * Groups
    Results = runFile(files[index+start], verbose, small, index)