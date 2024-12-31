import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.special as sp
import os as os
import multiprocessing
from joblib import Parallel, delayed
from time import time as ti
from skimage.restoration import denoise_wavelet
import pickle
import CoreFunctions as cf
import sys
import random
import psutil

LastSuccesfull = int(sys.argv[1])
DateString = sys.argv[2]
MakeOnesOrZeros = int(sys.argv[3])             # 1022 is the moving, 1011 is finishing stationary
ThirdTry = int(sys.argv[4])                    # no parallel if 1
FilesPerRun = int(sys.argv[5])                 # 1022 = 18, 1011 = 15
ConcurrentFiles = int(sys.argv[6])             # 1022 = 6, 1011 = 5
BadFile = int(sys.argv[7])                     # add if a bad file was seen on a previous run to do that as verbose
IssueMat = int(sys.argv[8])                    #if it is dieing on making a print from a matrix, last known good +1

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow import convert_to_tensor as ctt

DataFolder = '/sciclone/scr10/dchendrickson01/Recordings2/'
DataFolder = '/scratch/Recordings2/'
model_directory = '/scratch/models/stopped/'


TIME_STEPS = 1200
Skips = 125
RollSize = 50

tic = ti()
start = tic

MemoryProtection = True
noisy = True



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
    return np.asarray(StdDevs[:-RollSize].T)

def SquelchPattern(DataSet, StallRange = 5000, SquelchLevel = 0.02, verbose = noisy):
    
    SquelchSignal = np.ones(len(DataSet))
    if verbose:
        print(len(SquelchSignal))
        
    for i in range(len(DataSet)-2*StallRange):
        if np.average(DataSet[i:i+StallRange]) < SquelchLevel:
            SquelchSignal[i+StallRange]=0

    return SquelchSignal

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

def split_list_by_zeros(original_list, ones_list):
    # modified split_list_by_ones function to instead split by the zeros.
    #
    #
    # Created with Bing AI support
    #  1st request: "python split list into chunks based on value"
    #  2nd request: "I want to split the list based on the values in a second list.  Second list is all 1s and 0s.  I want all 0s removed, and each set of consequtive ones as its own item"
    #  3rd request: "That is close.  Here is an example of the two lists, and what I would want returned: original_list = [1, 2, 3, 8, 7, 4, 5, 6, 4, 7, 8, 9]
    #                ones_list =     [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    #                return: [[1, 2, 3, 8], [4, 5, 6], [8,9]]"
    #
    #This is the function that was created and seems to work on the short lists, going to use for long lists
    
    result_sublists = []
    sublist = []

    for val, is_one in zip(original_list, ones_list):
        if not is_one:
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

def runFile(file, verbose = noisy, small = False, index=0, start=ti()):
    noise = verbose
    if file[-4:] == '.csv':    
        dataset = pd.read_csv(DataFolder+file, delimiter =",", header=None, engine='python',on_bad_lines='skip')
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
        dataset['FracSec'].replace('',0)
        dataset.replace([np.nan, np.inf, -np.inf],0,inplace=True)
        
        dataset[['Day','Second']] = dataset[['Day','Second']].apply(lambda x: x.astype(int).astype(str).str.zfill(6))
        dataset[['FracSec']] = dataset[['FracSec']].apply(lambda x: x.astype(int).astype(str).str.zfill(4))

        dataset["timestamp"] = pd.to_datetime(dataset.Day+dataset.Second+dataset.FracSec,format='%y%m%d%H%M%S%f')
        dataset["timestamps"] = dataset["timestamp"]

        dataset["p"] = dataset.p - np.average(dataset.p)
        dataset["h"] = dataset.h - np.average(dataset.h)
        dataset["v"] = dataset.v - np.average(dataset.v)
        dataset["r"] = np.sqrt(dataset.p**2 + dataset.h**2 + dataset.v**2)

        dataset.index = dataset.timestamp

        dataset["SmoothP"] = denoise_wavelet(dataset.p, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        dataset["SmoothH"] = denoise_wavelet(dataset.h, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        dataset["SmoothV"] = denoise_wavelet(dataset.v, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        dataset["SmoothR"] = denoise_wavelet(dataset.r, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')

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

        SmoothDevZ[np.isnan(SmoothDevZ)]=0
        
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

        #df_pr = split_list_by_zeros(dataset.p, dataset.IsMoving)
        #df_hr = split_list_by_ones(dataset.h, dataset.IsMoving)
        #df_vr = split_list_by_ones(dataset.v, dataset.IsMoving)
        #df_rrr = split_list_by_ones(dataset.r, dataset.IsMoving)
        if MakeOnesOrZeros == 1:
            df_ps = split_list_by_ones(dataset.SmoothP, dataset.IsMoving)
            df_hs = split_list_by_ones(dataset.SmoothH, dataset.IsMoving)
            df_vs = split_list_by_ones(dataset.SmoothV, dataset.IsMoving)
            df_rs = split_list_by_ones(dataset.SmoothR, dataset.IsMoving)
        else:
            df_ps = split_list_by_zeros(dataset.SmoothP, dataset.IsMoving)
            df_hs = split_list_by_zeros(dataset.SmoothH, dataset.IsMoving)
            df_vs = split_list_by_zeros(dataset.SmoothV, dataset.IsMoving)
            df_rs = split_list_by_zeros(dataset.SmoothR, dataset.IsMoving)
            

        del dataset
        
        MatsSmooth = []
        for i in range(len(df_ps)):
            MatsSmooth.append(np.vstack((df_ps[i],df_hs[i],df_vs[i],df_rs[i])))
        
        if verbose:
            print("Split by ones", ti()-start)

        if verbose:
            print('format changed', ti()-start, len(MatsSmooth))

        return MatsSmooth
    else:
        return ['fail','fail']
        


def runWrapper(file_path, verbose=noisy, small=False, index=0, start=ti()):
    try:
        rtrn = runFile(file_path, verbose, small, index, start)
        return rtrn
    except Exception as e:
        with open('BadInputs.text', 'a') as bad_file:
            bad_file.write(file_path + '\n')
        return np.zeros((10, 10, 3))

def PrintWrap(Mat, Verbose = False):
    if Verbose:
        print('Starting Issue Mat')
    localPrints = []
    lenm = np.shape(Mat)[1]
    if Issue:
        print(f'Mat length is {lenm}')
    slices = int(lenm/TIME_STEPS)
    if Verbose:
        print(f'Slices needed is {slices}')
    for i in range(slices):
        temp = (cf.makeMPFast(Mat[:3,i*TIME_STEPS:(i+1)*TIME_STEPS], wvt = 'sym4', scales = 32, spacer = 2, title = ''))
        localPrints.append(temp.astype(np.float32)/255.0)
        if Verbose and i%10==0:
            print(f'Making number {i}')
    return localPrints

if LastSuccesfull ==0:
    files= os.listdir(DataFolder) 
    random.shuffle(files)
    with open(f'CurrentFileList{DateString}.text','w') as file:
        for item in files:
            file.write(f"{item}\n")
else:
    with open(f'CurrentFileList{DateString}.text','r') as file:
        files = file.readlines()
    files=[item.strip() for item in files]

toc=ti()

j=LastSuccesfull+1
Mats=[]
if ThirdTry == 0:
    AllDatas = Parallel(n_jobs=ConcurrentFiles,timeout=999)(delayed(runWrapper)(files[(j*FilesPerRun+i)], False, False, 0, ti()) for i in range(FilesPerRun))
else:
    AllDatas = []
    for i in range(FilesPerRun):
        if i == BadFile:
            localVerbose = True
        else:
            localVerbose = False
        AllDatas.append(runWrapper(files[(j*FilesPerRun+i)], localVerbose, False, 0, ti()))
        print(f'Wrapper {i} group {LastSuccesfull}, in {int((ti()-toc)/.6)/100} minutes')

for fileResponse in AllDatas:
    for Mat in fileResponse:
        Mats.append(Mat)

if MemoryProtection:
    del AllDatas
    print('RAM after AllData:', psutil.virtual_memory()[2])        
lengths = []
rejects = []
Keeps = []

for Mat in Mats:
    spm = np.shape(Mat)
    if len(spm) > 1:
        lenM = spm[1]
    else:
        lenM = 1
    if (lenM > 1250):
        lengths.append(lenM)
        Keeps.append(Mat)
    else:
        rejects.append(lenM)

if MemoryProtection:
    del Mats

Prints = []

if ThirdTry == 0:
    AllPrints = Parallel(n_jobs=ConcurrentFiles)(delayed(PrintWrap)(Mat) for Mat in Keeps)
else:
    AllPrints = []
    for i, Mat in enumerate(Keeps):
        if IssueMat != 0:
            AllPrints.append(PrintWrap(Mat, True))
        else:
            AllPrints.append(PrintWrap(Mat))
        print(f'Prints {i} of {len(Keeps)}, in {int((ti()-toc)/.6)/100} minutes')



if MemoryProtection:
    del Keeps
    print('RAM after Keeps:', psutil.virtual_memory()[2])
for group in AllPrints:
    for fprint in group:
        Prints.append(fprint[:, ::2, :])

if MemoryProtection:
    del AllPrints

random.shuffle(Prints)

for i, image in enumerate(Prints):
    if not isinstance(image, np.ndarray):
        Prints[i] = np.array(image, dtype=np.float32)
    elif image.dtype != np.float32:
        Prints[i] = image.astype(np.float32)

# Stack the images into a single NumPy array
prints_array = np.stack(Prints, axis=0)

if MemoryProtection:
    del Prints
    print('RAM after Prints:', psutil.virtual_memory()[2])
# Convert the NumPy array to a TensorFlow tensor

trX = ctt(prints_array)

if MemoryProtection:
    del prints_array

if MakeOnesOrZeros == 1:
    MoveStationary = 'Moving'
else:
    MoveStationary = 'Stationary'

print(f'Now Saving the file: {MoveStationary}Dataset_{str(j).zfill(4)}_{str(trX.shape[0]).zfill(6)}')
    
with open(DataFolder + f'MLPickles/{MoveStationary}Dataset_{str(j).zfill(4)}_{str(trX.shape[0]).zfill(6)}.p', 'wb') as handle:
    pickle.dump(trX, handle)

if MemoryProtection:
    del trX

print(f'{j} Done in {int((ti()-toc)/.6)/100} minutes. Using { psutil.virtual_memory()[2]} of RAM')
