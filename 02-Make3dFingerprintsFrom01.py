#!/sciclone/home20/dchendrickson01/.conda/envs/tfcgpu/bin/python


#Standard Header used on the projects
# %%

dataSize = 'big' #'big'  # 'small'

#first the major packages used for math and graphing
import numpy as np

import os as os
import random
import multiprocessing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#import cv2
from sklearn.model_selection import train_test_split

import datetime

import CoreFunctions as cf
import pickle

import platform

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
    location = '/sciclone/home20/dchendrickson01/'
elif Computer == "WinLap":
    location = 'C:\\Data\\'
elif Computer == "Desktop":
    location = "E:\\Backups\\Dan\\CraneData\\"
elif Computer == "LinLap":
    location = '/home/dan/Output/'
    

if Computer ==  "SciClone":
    rootfolder = '/sciclone/home20/dchendrickson01/'
    if dataSize == 'big':
        folder = '/sciclone/scr10/dchendrickson01/RecordingsSplit/xFold/'
        imFolder ='/sciclone/scr10/dchendrickson01/RecordingsSplit/750ptDB3/'
    else:
        folder = '/sciclone/data10/dchendrickson01/SmallCopy/'
        imFolder = '/sciclone/data10/dchendrickson01/SmallCopy/'
elif Computer == "Desktop":
    rootfolder = location
    imFolder = "E:\\Backups\\Dan\\CraneData\\Images\\"
    if dataSize == 'big':
        folder = 'G:\\CraneData\\'
    else:
        folder = rootfolder + "SmallCopy\\"
elif Computer =="WinLap":
    rootfolder = location
    folder = rootfolder + "SmallCopy\\"   
elif Computer == "LinLap":
    rootfolder = '/home/dan/Data/'
    folder = rootfolder + 'SmallCopy/'
    
scales = 100
#img_height , img_width = scales, 200
DoSomeFiles = False

SmoothType = 0  # 0 = none, 1 = rolling average, 2 = low pass filter, 3 = Kalman filter
WaveletToUse = 'db3'

num_cores = multiprocessing.cpu_count() -1
NumberOfFiles = num_cores - 2
GroupSize = NumberOfFiles


files = os.listdir(folder)

files=files[::-1]

if DoSomeFiles: files = random.sample(files,NumberOfFiles*2)

def GetPickleData(file,j):
    ff = open(folder+file,'rb')
    dump = pickle.load(ff)
    
    MoveSegments = dump[0]
    MoveName = dump[1]
    
    del dump
    
    for i in range(np.shape(MoveSegments)[0]):
        FP = cf.makeMPFast(MoveSegments[i,:,:].T,WaveletToUse, scales)
        FP = np.flip(FP,axis=0)
        
        fig  = plt.figure()
        plt.imshow(FP)
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.savefig(imFolder+MoveName+'-'+str(i).zfill(6)+'.png',bbox_inches='tight')
        plt.close()
    
    print(j,end=', ')
    
    return MoveName, j

Results = Parallel(n_jobs=num_cores)(delayed(GetPickleData)(files[i], i) for i in range(len(files)))