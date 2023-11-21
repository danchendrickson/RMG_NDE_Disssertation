#!/sciclone/home20/dchendrickson01/.conda/envs/tfcgpu/bin/python

print('Started')

#Standard Header used on the projects
# %%

dataSize = 'big'  # 'small'

#first the major packages used for math and graphing
import numpy as np

import os as os
import random
import multiprocessing
from joblib import Parallel, delayed

import cv2
from sklearn.model_selection import train_test_split

import datetime

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
        folder = '/sciclone/scr10/dchendrickson01/CraneData/'
        imFolder ='/sciclone/scr10/dchendrickson01/BigData/'
    else:
        folder = '/sciclone/data10/dchendrickson01/SmallCopy/'
elif Computer == "Desktop":
    rootfolder = location
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
    

scales = 500
#img_height , img_width = scales, 200
DoSomeFiles = False

SmoothType = 3  # 0 = none, 1 = rolling average, 2 = low pass filter, 3 = Kalman filter
WaveletToUse = 'beta'

num_cores = multiprocessing.cpu_count() -1
NumberOfFiles = num_cores - 2
GroupSize = NumberOfFiles


files = os.listdir(folder)
if DoSomeFiles: files = random.sample(files,NumberOfFiles)

import CoreFunctions as cf

def resizeImage(FP):
    res = cv2.resize(FP, dsize=(int(np.shape(FP)[0]/2), int(np.shape(FP)[1]/6)), interpolation=cv2.INTER_CUBIC)

    return res

def saveImage(FP, FName):
    cv2.imwrite(imFolder + FName + '.png', FP)
    return 1

def MakeImageFiles(files):
    numF = np.size(files)
    Keep = np.zeros(numF)
    for i in range(numF):
        if os.path.isfile(imFolder + files[i] + '.png'):
            Keep[i]=1
    Keep = np.array(Keep, dtype=int)   
    files=np.array(files)[Keep]
    
    AllAccels = Parallel(n_jobs=num_cores)(delayed(cf.getAcceleration)(file) for file in files)
    Flattened = []
    for j in range(np.shape(AllAccels)[0]):
        if AllAccels[j][0] == False:
            print(j,AllAccels[j][1])
        else: 
            Flattened.append(AllAccels[j])

    MetaData = []  #np.asarray([],dtype=object)
    DataOnlyMatrix = np.asarray([],dtype=object)
    for j in range(np.shape(AllAccels)[0]):
        if AllAccels[j][0] == False or np.shape(AllAccels[j][0][2])[0] != 60000:
            if AllAccels[j][1][4:9] =='Accel':
                print(j,AllAccels[j][1])
        else: 
            for k in range(3):
                MetaData.append([AllAccels[j][k][0], AllAccels[j][k][1], AllAccels[j][k][3], AllAccels[j][k][4]])
                if np.size(DataOnlyMatrix) == 0:
                        DataOnlyMatrix =np.matrix(AllAccels[j][k][2])
                else:
                        DataOnlyMatrix = np.concatenate((DataOnlyMatrix,np.matrix(AllAccels[j][k][2])),axis=0)

    MetaData = np.matrix(MetaData)

    AllAccels = cf.KalmanGroup(DataOnlyMatrix)

    del DataOnlyMatrix

    maxes = np.amax(AllAccels[:,500:], axis = 1)
    mins = np.amin(AllAccels[:,500:], axis = 1)

    Keep = np.zeros(mins.size)
    for i in range(mins.size):
        if i % 3 == 0:
            if maxes[i] > 0.01 and mins[i] < -0.01:
                Keep[i]=1
                Keep[i+1]=1
                Keep[i+2]=1
                #print(i)


    Keep = np.array(Keep, dtype='bool')

    AllAccels = AllAccels[Keep,:]
    MetaData = MetaData[Keep,:]

    MotionsLeft = int(np.shape(AllAccels)[0]/3.0)

    AllFingers =  Parallel(n_jobs=num_cores)(delayed(cf.makeMatrixImages)([AllAccels[i*3],AllAccels[i*3+1],AllAccels[i*3+2]]) for i in range(MotionsLeft))
    del AllAccels

    SmallFingers =  Parallel(n_jobs=num_cores)(delayed(resizeImage)(FP) for FP in AllFingers)
    del AllFingers

    #count =  Parallel(n_jobs=num_cores)(delayed(saveImage)(SmallFingers[i], MetaData[i*3,3]) for i in range(MotionsLeft))
    for i in range(MotionsLeft):
        count=saveImage(SmallFingers[i], MetaData[i*3,3])

    return sum(count)

GroupSize = NumberOfFiles



fCount = len(files)
GroupsLeft = int(fCount/GroupSize) + 1

SplitRatio = 1/(GroupsLeft)

RemainingFiles, GroupFiles, x,y = train_test_split(files, range(len(files)), test_size=SplitRatio, shuffle=True, random_state=0)

GroupsLeft -=1

count = MakeImageFiles(GroupFiles)

starttime = datetime.datetime.now()
looptime = starttime
i = 1
while GroupsLeft > 0:
    SplitRatio = 1/(GroupsLeft)

    RemainingFiles, GroupFiles, x,y = train_test_split(RemainingFiles, range(len(RemainingFiles)), test_size=SplitRatio, shuffle=True, random_state=0)
       
    GroupsLeft -=1

    count = MakeImageFiles(GroupFiles)
    #saver.restore('model.ckpt')
    tNow = datetime.datetime.now()
    
    print(count,i,GroupsLeft, tNow-starttime, tNow-looptime)
        #saver.save(sess,'model.ckpt')
    i+=1
    looptime = tNow

