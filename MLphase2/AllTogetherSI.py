# %% [markdown]
# # All Together Now
# ## Fingerprints for many wavelets, clustering, then sorting
# 
# Combines the code from 10, 11, 12.  Temporary in Jupyter Notebook, probably going to be converted to .py so it can run headless once it is trustworthy.  Tested on 3 files, going to go to 16.  Will do same stack, 2 cranes before and after tamping, 4 days of each set.

# %%
#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.special as sp

#Standard cycle to make black and white images and dashed and line styles
default_cycler = (cycler('color', ['0.00', '0.40', '0.60', '0.70']) + cycler(linestyle=['-', '-', '-', '-']))
plt.rc('axes', prop_cycle=default_cycler)
my_cmap = plt.get_cmap('gray')

# %%
#Wavelet Imports Extra Headers:
import os as os
import pywt as py
import statistics as st
import os as os
import random
import multiprocessing
from joblib import Parallel, delayed
import platform

from time import time as ti


# %%
import CoreFunctions as cf
from skimage.restoration import denoise_wavelet
import pandas as pd

# %%
# ML Imports Imports
#from keras.preprocessing import image
import keras.utils as image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np
from sklearn.cluster import KMeans
import os, shutil, glob, os.path
from PIL import Image as pil_image
image.LOAD_TRUNCATED_IMAGES = True 


# %%
import pandas as pd

# %% [markdown]
#  ## Choosing Platform
#  Working is beinging conducted on several computers, and author needs to be able to run code on all without rewriting..  This segment of determines which computer is being used, and sets the directories accordingly.

# %%
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


# %%
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


# %%
if Computer ==  "SciClone":
    rootfolder = '/sciclone/home/dchendrickson01/'
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


# %%
from contextlib import contextmanager
import sys, os

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

# %% [markdown]
# ## Set up variables

# %%
# Maunally chooseing before and after tamping for same track

files = ['221206 recording1.csv','221207 recording1.csv','221208 recording1.csv','221209 recording1.csv',
         '221206 recording2.csv','221207 recording2.csv','221208 recording2.csv','221209 recording2.csv',
         '230418 recording1.csv','230419 recording1.csv','230420 recording1.csv','230421 recording1.csv',
         '230418 recording2.csv','230419 recording2.csv','230420 recording2.csv','230421 recording2.csv']


# %%
ClustersWanted = 11
scales= 100
skips = 1
minLength = 750

subfolder ='wvltSort/'
#subfolder = 'scaleSort/'


# %% [markdown]
# ## Project Specific Functions

# %%
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

def RollingSum(Data, Length = 100):
    RollSumStdDev = []
    for i in range(Length):
        RollSumStdDev.append(sum(Data[0:i+1]))
    for i in range(len(Data) - Length):
        RollSumStdDev.append(sum(Data[i:i+Length]))
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
            if Timestamps[j] > Timestamps[i]:
                dTime[j]=Timestamps[j]-Timestamps[i]
            else:
                dTime[j]=Timestamps[j]-Timestamps[i]+10000.0
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



# %%
#Smooth = cf.Smoothing(ODataSet[:,3],2) #,50)
def DeviationVelocity(file):
    if file[-3:] =='csv':
        ODataSet = np.genfromtxt(open(folder+file,'r'), delimiter=',',skip_header=0,missing_values=0,invalid_raise=False)
        SmoothX = denoise_wavelet(ODataSet[:,3], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SmoothY = denoise_wavelet(ODataSet[:,4], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SmoothZ = denoise_wavelet(ODataSet[:,5], method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SmoothX -= np.average(SmoothX)
        SmoothY -= np.average(SmoothY)
        SmoothZ -= np.average(SmoothZ)
        StdDevsX = RollingStdDev(ODataSet[:,3],SmoothX)
        StdDevsX.append(0)
        StdDevsX = np.asarray(StdDevsX)
        SmoothDevX = denoise_wavelet(StdDevsX, method='VisuShrink', mode='soft', wavelet_levels=3, wavelet='sym2', rescale_sigma='True')
        SquelchSignal = SquelchPattern(SmoothDevX, 2000, 0.03)
        #Velocity = getVelocity(ODataSet[:,3], ODataSet[:,2],SquelchSignal, 2)
        #Velocity = np.asarray(Velocity)
        MoveMatrix = np.matrix([SmoothX, SmoothY, SmoothZ])
        return [SquelchSignal,MoveMatrix,SmoothDevX,file[:-3]]
    else:
        pass


# %%
def SepreateMovements(SquelchSignal, RawData, FileName):
    Moves= []
    MoveNames = []
    Move = np.zeros((1,3), dtype=float)
    i = 0
    for j in range(len(SquelchSignal)-1):
        if SquelchSignal[j] == 1:
            try:
                Move = np.concatenate((Move, RawData[j,:]), axis=0)
            except:
                print(j)
            if SquelchSignal[j+1] == 0:
                #Move = np.matrix(Move)
                Moves.append(Move)
                MoveNames.append(FileName + str(i).zfill(3))
                i+=1
                Move = np.zeros((1,3), dtype=float)
                #Move[0,2]=0
    Moves.append(Move)
    MoveNames.append(FileName + str(i).zfill(3))
    return Moves, MoveNames
    


# %%
def splitLong(Moves, maxLength = 4000, minLength = 1000, MoveNames = []):
    if len(MoveNames) <=1:
        MoveNames = ['null'  for x in range(len(Moves))]
    Xmoves = []
    Xnames = []
    for i in range(len(Moves)):
        if np.shape(move)[0] > maxLength: 
            Xmoves.append(Moves[i][:int(len(Moves[i])/2),:])
            Xnames.append(MoveNames[i] + 'a')
            Xmoves.append(Moves[i][int(len(Moves[i])/2):,:])
            Xnames.append(MoveNames[i] + 'b')
        else:
            if np.shape(Moves[i])[0] < minLength:
                pass
            else:
                Xmoves.append(Moves[i])
                Xnames.append(MoveNames[i])
    return Xmoves, Xnames

def findMaxLength(Moves):
    maxLength = 0
    LongMove = 0
    for i in range(len(Moves)):
        if np.shape(Moves[i])[0] > maxLength: 
            maxLength =  np.shape(Moves[i])[0]
            LongMove = i
    return maxLength, LongMove

def findMinLength(Moves):
    minLength = 9999999
    SmallMove = 0
    for i in range(len(Moves)):
        if np.shape(Moves[i])[0] < minLength: 
            minLength =  np.shape(Moves[i])[0]
            SmallMove = i
    return minLength, SmallMove



# %%
import scipy as ss
def MakeSpectrogramImages(data, title, something=300, nperseg = 512, novrelap=256, folder=imageFolder):
    f, t, Szz = ss.signal.spectrogram(data,something,nperseg = nperseg, noverlap=novrelap)
    fig = plt.figure(figsize=(8,3), dpi=800)
    ax = plt.axes()
    ax.set_axis_off()
    plt.pcolormesh(t, f, Szz[0],cmap='gist_ncar')
    plt.savefig(folder+'spec/'+title+'.png',bbox_inches='tight', pad_inches=0)


# %%


def sortClusters(folder):
    
    filelist = glob.glob(os.path.join(folder, '*.png'))
    filelist.sort()
    
    if len(filelist) > 10:
    
        with suppress_stdout():
            model = VGG16(weights='imagenet', include_top=False)

        sampleName = folder.split('/')[-2]
        print(sampleName)

        if os.path.exists(imageFolder+subfolder+sampleName+'/') == False:
            os.mkdir(imageFolder+subfolder+sampleName+'/')

        # Variables
        imdir = folder # DIR containing images
        targetdir =imageFolder+subfolder+sampleName+'/' # DIR to copy clustered images to
        number_clusters = ClustersWanted

        # Loop over files and get features

        featurelist = []
        for i, imagepath in enumerate(filelist):
            #try:
            #if i %100 == 0 : print("    Status: %s / %s" %(i, len(filelist)), end="\r")
            img = image.load_img(imagepath, target_size=(224, 448))
            img_data = image.img_to_array(img)
            img_data = np.expand_dims(img_data, axis=0)
            img_data = preprocess_input(img_data)
            with suppress_stdout():
                features = np.array(model.predict(img_data))
            featurelist.append(features.flatten())
            #except:
            #    continue

            # Clustering
        kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(np.array(featurelist))

        # Copy images renamed by cluster 
        # Check if target dir exists
        try:
            os.makedirs(targetdir)
        except OSError:
            pass
        # Copy with cluster name

        for i, m in enumerate(kmeans.labels_):
            try:
                shutil.copy(filelist[i], targetdir + str(m) + "_" + filelist[i].split('/')[-1])
            except:
                continue

    return 1

# %% [markdown]
# ## Process Files
# 

# %%
LoopFiles = 3
loops = int(len(files) / LoopFiles) 
if len(files)%LoopFiles != 0:
    loops += 1


# %%
SquelchSignal = []
RawData=[]
OrderedFileNames=[]


# %%

st = ti()

for k in range(loops):
    if k == loops -1:
        tfiles = files[k*LoopFiles:]
    else:
        tfiles = files[k*LoopFiles:(k+1)*LoopFiles]
    Results = Parallel(n_jobs=LoopFiles)(delayed(DeviationVelocity)(file) for file in tfiles)
    
    for i in range(len(Results)):       
        SquelchSignal.append(Results[i][0])
        RawData.append(np.matrix(Results[i][1]).T)
        OrderedFileNames.append(Results[i][3])
    print(k, np.shape(Results), (ti()-st)/60.0)
    



# %%
MoveData = Parallel(n_jobs=31)(delayed(SepreateMovements)(SquelchSignal[i], RawData[i], OrderedFileNames[i])
                                       for i in range(len(RawData)))

# %%
Movements = []
GroupNames = []
for move in MoveData:
    Movements.append(move[0])
    GroupNames.append(move[1])



# %%
Moves=[]
for Groups in Movements:
    for Move in Groups:
        Moves.append(Move)

MoveNames = []
for Groups in GroupNames:
    for name in Groups:
        MoveNames.append(name)

# %%

del SquelchSignal
del RawData
del Movements
del GroupNames
del MoveData
del OrderedFileNames


# %%
longMove, MoveNumb = findMaxLength(Moves)


# %%

WvltFam = py.families()
Wvlts = []
for Fam in WvltFam:
    temp = py.wavelist(Fam)
    for wvlt in temp:
        Wvlts.append(wvlt)
        
trys = Wvlts

trys.append('beta')


# %%
Moves, MoveNames = splitLong(Moves, longMove+1, minLength, MoveNames)


# %%
#StorageFolder = imageFolder + 'wvltTest/'
StorageFolder = imageFolder + 'scaleTest/'

# %%
f = 0

#wvlt = 'beta'
'''
for tri in trys:

    if os.path.exists(StorageFolder+tri+'/') == False:
        os.mkdir(StorageFolder+tri+'/')

    #FPimages = Parallel(n_jobs=60)(delayed(cf.makeMPFast)(Moves[i].T, tri, scales, skips,StorageFolder+tri + '/Move '+ MoveNames[i]) for i in range(len(Moves)))
    FPimages = Parallel(n_jobs=60)(delayed(cf.makeMatrixImages)(Moves[i].T, tri, scales, skips,StorageFolder+tri + '/Move '+ MoveNames[i]) for i in range(len(Moves)))
##FPimages = Parallel(n_jobs=60)(delayed(cf.makeMPFast)(Moves[MoveNum].T, tri, scales, skips, StorageFolder+'wvltTest/' + tri + '_LongMove') for tri in trys)


# %%
if os.path.exists(StorageFolder+'spec/') == False:
        os.mkdir(StorageFolder+'spec/')

FPimages = Parallel(n_jobs=60)(delayed(MakeSpectrogramImages)(Moves[i].T, 'Move '+ MoveNames[i], 300, 512, 505, StorageFolder) for i in range(len(Moves)))


# %%
del FPimages
del Moves
'''

# %% [markdown]
# ## Started the Unsupervised Clustering

# %%
folders = glob.glob(StorageFolder + '*/')
print(len(folders), StorageFolder)

# %%
MaxSameTime = 2
J = int(len(folders) / MaxSameTime)
if len(folders) % MaxSameTime != 0:
    J +=1
    #sorting = Parallel(n_jobs=3)(delayed(sortClusters)(folder) for folder in folders[53:59])
for j in range(J):
    if j!=J:
        sorting = Parallel(n_jobs=3)(delayed(sortClusters)(folder) for folder in folders[MaxSameTime * j:MaxSameTime*(j+1)])
    else:
        sorting = Parallel(n_jobs=3)(delayed(sortClusters)(folder) for folder in folders[MaxSameTime * j:])
    print(str(j) + ' of ' + str(J))

# %% [markdown]
# ## Now for comaprison of the results

# %%
def GetResultsDataFrame(folder):
    files = os.listdir(folder)
    if len(files) > 2:
        TypeName = folder.split('/')[-2]
        Results = []
        for file in files:
            Group = int(file.split('_')[0])
            Move = file.split('_')[1][5:-4]
            Results.append([Group,Move])
        Results = np.matrix(Results)
        temp_dict = {
                "MoveName" : np.asarray(Results[:,1]).flatten(),
                TypeName: np.asarray(Results[:,0]).flatten()
            }
        DataSet = pd.DataFrame(temp_dict)
        return DataSet

# %%
targetdir = imageFolder+'scaleSort/'
#targetdir = imageFolder+'wvltSort/'

folders = glob.glob(targetdir+'*/')

# %%
RealReturns = False
for folder in folders:
    FolderDF = GetResultsDataFrame(folder)
    #print(folder.split('/')[-2], len(FolderDF))
    if len(FolderDF) > 2:
        if RealReturns == True:
            AllData = pd.merge(AllData, FolderDF, on ='MoveName', how ="outer")
        else:
            AllData = FolderDF
            RealReturns = True

# %%
#AllData.to_csv('ClusteredResults2.csv')
AllData.to_csv('ScalesResults.csv')


