#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
 
from cycler import cycler
import scipy.special as sp
import matplotlib.pyplot as plt


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

#import CoreFunctions as cf
#from skimage.restoration import denoise_wavelet

#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
 
from cycler import cycler
import scipy.special as sp
import matplotlib.pyplot as plt


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

#import CoreFunctions as cf
from skimage.restoration import denoise_wavelet

from typing import List, Optional, Callable, Iterable
from itertools import islice

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates as mdates

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
    location = '/sciclone/home/dchendrickson01/image/'
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
    folder = '/scratch/Recordings2/'
    imageFolder = '/scratch/Move3Dprint/'
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
freq = "2s"
prediction_length=50

files = ['230418 recording1.csv','230419 recording1.csv','230420 recording1.csv','230421 recording1.csv',
         '230418 recording2.csv','230419 recording2.csv','230420 recording2.csv','230421 recording2.csv']

def RollingStdDev(RawData, SmoothData, RollSize = 25):
    StdDevs = []
    for i in range(RollSize):
        Diffs = RawData.iloc[0:i+1]-SmoothData.iloc[0:i+1]
        Sqs = Diffs * Diffs
        Var = sum(Sqs) / (i+1)
        StdDev = np.sqrt(Var)
        StdDevs.append(StdDev)
    for i in range(len(RawData)-RollSize-1):
        j = i + RollSize
        Diffs = RawData.iloc[i:j]-SmoothData.iloc[i:j]
        Sqs = Diffs * Diffs
        Var = sum(Sqs) / RollSize
        StdDev = np.sqrt(Var)
        StdDevs.append(StdDev)

    return StdDevs

def RollingSum(Data, Length = 100):
    RollSumStdDev = []
    for i in range(Length):
        RollSumStdDev.append(sum(Data[0:i+1]))
    for i in range(len(Data) - Length):
        RollSumStdDev.append(sum(Data.iloc[i:i+Length]))
    return RollSumStdDev

def SquelchPattern(DataSet, StallRange = 5000, SquelchLevel = 0.02):
    SquelchSignal = np.ones(len(DataSet))

    for i in range(len(DataSet)-2*StallRange):
        if np.average(DataSet[i:i+StallRange]) < SquelchLevel:
            SquelchSignal[i+StallRange]=0

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
            if Timestamps.iloc[j] > Timestamps.iloc[i]:
                dTime[j]=Timestamps.iloc[j]-Timestamps.iloc[i]
            else:
                dTime[j]=Timestamps.iloc[j]-Timestamps.iloc[i]+10000.0
        dTime /= 10000.0

    velocity[0] = Acceleration.iloc[0] * (dTime[0])

    for i in range(len(Acceleration)-1):
        j = i + 1
        if corrected ==2:
            if Squelch.iloc[j]==0:
                velocity[j]=0
            else:
                velocity[j] = velocity[i] + Acceleration.iloc[j] * dTime[j]                
        else:
            velocity[j] = velocity[i] + Acceleration.iloc[j] * dTime[j]

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

def MakeDataframe(file):
    dataset = pd.read_table(folder+file, delimiter =", ", header=None, engine='python')

    dataset = dataset.rename(columns={0:"Day"})
    dataset = dataset.rename(columns={1:"Second"})
    dataset = dataset.rename(columns={2:"PartSec"})
    dataset = dataset.rename(columns={3:"p"})
    dataset = dataset.rename(columns={4:"h"})
    dataset = dataset.rename(columns={5:"v"})
    dataset = dataset.rename(columns={6:"Sensor"})

    dataset[['Day','Second']] = dataset[['Day','Second']].apply(lambda x: x.astype(int).astype(str).str.zfill(6))
    dataset[['FracSec']] = dataset[['PartSec']].apply(lambda x: x.astype(int).astype(str).str.zfill(4))

    dataset["timestamp"] = pd.to_datetime(dataset.Day+dataset.Second
                                          +dataset.FracSec, format='%y%m%d%H%M%S%f')
    dataset["dt"]=dataset.timestamp - dataset.timestamp.shift(1, fill_value=dataset.timestamp[0])
    
    dataset["timestamps"] = pd.Series(pd.date_range(dataset.timestamp.dt.date[0], periods=len(dataset),freq=dataset.dt.mean())).dt.time
    
    

    dataset["p"] = dataset.p - np.average(dataset.p)
    dataset["h"] = dataset.h - np.average(dataset.h)
    dataset["v"] = dataset.v - np.average(dataset.v)
    dataset["r"] = np.sqrt(dataset.p**2 + dataset.h**2 + dataset.v**2)

    dataset.index = dataset.timestamps

    dataset["SmoothV"] = denoise_wavelet(dataset.v, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')

    StdDevsV = RollingStdDev(dataset.v, dataset.SmoothV)
    StdDevsV.append(0)
    StdDevsV = np.asarray(StdDevsV)
    SmoothDevV = denoise_wavelet(StdDevsV, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')

    Max = np.max(SmoothDevV)
    buckets = int(Max / 0.005) + 1
    bins = np.linspace(0,buckets*0.005,buckets+1)
    counts, bins = np.histogram(SmoothDevV,bins=bins)

    CummCount = 0
    HalfWay = 0
    for i in range(len(counts)):
        CummCount += counts[i]
        if CummCount / len(SmoothDevV) >= 0.5:
            if HalfWay == 0:
                HalfWay = i

    SquelchLevel = bins[HalfWay] 
    dataset["IsMoving"] = SquelchPattern(SmoothDevV, 4000, SquelchLevel)

    dataset["velocity"] = getVelocity(dataset.r, dataset.PartSec, dataset.IsMoving, 2)
    
    dataset.timestamps.apply(pd.Timestamp.strip('[]'))
    
    return dataset

print('getting data')
data = Parallel(n_jobs=16)(delayed(MakeDataframe)(file) for file in files)
#data=MakeDataframe(files[0])
dataset = pd.concat(data)[["p","h","v","r","IsMoving","velocity","timestamps","timestamp"]]
dataset.head()
print('Have data')

from gluonts.torch import DeepAREstimator
#from gluonts.torch.trainer import Trainer
from gluonts.evaluation import make_evaluation_predictions, Evaluator


def train_and_predict(dataset, estimator):
    predictor = estimator.train(dataset)
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset, predictor=predictor
    )
    evaluator = Evaluator(quantiles=(np.arange(20) / 20.0)[1:])
    agg_metrics, item_metrics = evaluator(ts_it, forecast_it, num_series=len(dataset))
    return agg_metrics["MSE"]

estimator = DeepAREstimator(
    freq=freq, prediction_length=prediction_length
)

print('Model set up')

#dataset.timestamps.apply(pd.Timestamp.strip('[]'))


from gluonts.dataset.pandas import PandasDataset

#ds = PandasDataset(dataset, target="x", freq=freq)
ds = PandasDataset.from_long_dataframe(
    dataframe=dataset,
    item_id="timestamps",
    #timestamp="timestamps",
    freq="ns",
)

train_and_predict(ds, estimator)
