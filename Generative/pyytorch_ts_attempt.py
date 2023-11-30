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

#import CoreFunctions as cf
#from skimage.restoration import denoise_wavelet

import pandas as pd

import torch
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
freq = "2s"

files = ['230418 recording1.csv','230419 recording1.csv']#,'230420 recording1.csv','230421 recording1.csv',
         #'230418 recording2.csv','230419 recording2.csv','230420 recording2.csv','230421 recording2.csv']
    
def MakeDataframe(file):
    dataset = pd.read_table(folder+file, delimiter =", ", header=None, engine='python')

    dataset = dataset.rename(columns={0:"Day"})
    dataset = dataset.rename(columns={1:"Second"})
    dataset = dataset.rename(columns={2:"FracSec"})
    dataset = dataset.rename(columns={3:"x"})
    dataset = dataset.rename(columns={4:"y"})
    dataset = dataset.rename(columns={5:"z"})
    dataset = dataset.rename(columns={6:"Sensor"})

    dataset[['Day','Second']] = dataset[['Day','Second']].apply(lambda x: x.astype(int).astype(str).str.zfill(6))
    dataset[['FracSec']] = dataset[['FracSec']].apply(lambda x: x.astype(int).astype(str).str.zfill(4))

    dataset["timestamp"] = pd.to_datetime(dataset.Day+dataset.Second+dataset.FracSec,format='%y%m%d%H%M%S%f')

    dataset["x"] = dataset.x - np.average(dataset.x)
    dataset["y"] = dataset.y - np.average(dataset.y)
    dataset["z"] = dataset.z - np.average(dataset.z)
    dataset["r"] = np.sqrt(dataset.x**2 + dataset.y**2 + dataset.z**2)

    dataset.index = dataset.timestamp
    
    return dataset

data = Parallel(n_jobs=8)(delayed(MakeDataframe)(file) for file in files)
dataset = pd.concat(data)[["x","y","z","r"]]
dataset.head()

from gluonts.dataset.split import OffsetSplitter, DateSplitter

splitter = DateSplitter(
    date=pd.Period('2023-04-18', freq='3s'))

train, test_template = splitter.split(dataset)

test_dataset = test_template.generate_instances(
    prediction_length=7,
    windows=2,
    distance=3, # windows are three time steps apart from each other
)


evaluator = MultivariateEvaluator(quantiles=(np.arange(20)/20.0)[1:],
                                  target_agg_funcs={'sum': np.sum})


estimator = TempFlowEstimator(
    target_dim=4,
    prediction_length=test_dataset.prediction_length,
    cell_type='GRU',
    input_size=552,
    freq="1T",
    scaling=True,
    dequantize=True,
    n_blocks=4,
    trainer=Trainer(device=device,
                    epochs=45,
                    learning_rate=1e-3,
                    num_batches_per_epoch=100,
                    batch_size=64)
)

predictor = estimator.train(train)
forecast_it, ts_it = make_evaluation_predictions(dataset=dataset_test,
                                             predictor=predictor,
                                             num_samples=100)
forecasts = list(forecast_it)
targets = list(ts_it)

agg_metric, _ = evaluator(targets, forecasts, num_series=len(dataset_test))

torch.save(estimator.state_dict(), 'Something')
