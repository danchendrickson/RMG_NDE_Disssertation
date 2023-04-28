
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

# %%
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

# %%
import CoreFunctions as cf
from skimage.restoration import denoise_wavelet

# %% [markdown]
# ## Choosing Platform
# Working is beinging conducted on several computers, and author needs to be able to run code on all without rewriting..  This segment of determines which computer is being used, and sets the directories accordingly.

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
    location = '/sciclone/home20/dchendrickson01/image/'
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

# %%
#files = os.listdir(folder)
#files=files[39:41]

# %%
Saving = False
location = folder
Titles = True
Ledgends = True

f = 0


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
#files = fi2

# %%
# Maunally chooseing before and after tamping for same track

files = ['230103 recording3.csv','230104 recording3.csv','230105 recording3.csv'] #,'230106 recording3.csv',
#         '230103 recording4.csv','230104 recording4.csv','230105 recording4.csv','230106 recording4.csv']

# %%
LoopFiles = 3
loops = int(len(files) / LoopFiles) 
if len(files)%LoopFiles != 0:
    loops += 1

# %%
SquelchSignal = []
RawData=[]
OrderedFileNames=[]

st = ti()

# %%
for k in range(loops):
    if k == loops -1:
        tfiles = files[k*LoopFiles:]
    else:
        tfiles = files[k*LoopFiles:(k+1)*LoopFiles]
    Results = Parallel(n_jobs=LoopFiles)(delayed(DeviationVelocity)(file) for file in tfiles)
    
    for i in range(len(Results)):       
        SquelchSignal.append(Results[i][0])
        RawData.append(np.matrix(Results[i][1]).T)
        OrderedFileNames.append(Results[i][2])
    print(k, np.shape(Results), (ti()-st)/60.0)
    


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
longMove, MoveNumb = findMaxLength(Moves)

# %%
scales= 100
skips = 1
wvlt = 'coif'
trys = ['mexh','gaus2','dmey','gaus1','morl','cgau1','cgau2'] #'sym1','sym2','sym3','db1','db2']
wvlt = 'beta'

WvltFam = py.families()
Wvlts = []
for Fam in WvltFam:
    temp = py.wavelist(Fam)
    for wvlt in temp:
        Wvlts.append(wvlt)
        
trys = Wvlts

# %%
def MaxSpectrogram(Signal,i=0,Save=True):
    
    plt.specgram(Signal, Fs=6, cmap="rainbow")

    # Set the title of the plot, xlabel and ylabel
    # and display using show() function
    plt.title('Spectrogram Using matplotlib.pyplot.specgram() Method')
    plt.xlabel("DATA")
    plt.ylabel("TIME")
    if Save: plt.savefig(imageFolder+'specgram/Move '+str(i).zfill(4)+'.png')
    plt.show()
    
    return 0

# %%
Moves, MoveNames = splitLong(Moves, longMove+1, MoveNames)

# %%
f = 0

#wvlt = 'beta'

for tri in trys:

    os.mkdir(imageFolder+tri+'/')

    FPimages = Parallel(n_jobs=60)(delayed(cf.makeMPFast)(Moves[i].T, wvlt, scales, skips,imageFolder+'try/Move '+ MoveNames[i]) for i in range(len(Moves)))

##FPimages = Parallel(n_jobs=60)(delayed(cf.makeMPFast)(Moves[MoveNum].T, tri, scales, skips, imageFolder+'wvltTest/' + tri + '_LongMove') for tri in trys)

# %%
import scipy as ss
def MakeSpectrogramImages(data, title, something=300, nperseg = 512, novrelap=256):
    f, t, Szz = ss.signal.spectrogram(Moves[MoveNum][:,0].T,something,nperseg = nperseg, noverlap=novrelap)
    fig = plt.figure(figsize=(8,3), dpi=800)
    ax = plt.axes()
    ax.set_axis_off()
    plt.pcolormesh(t, f, Szz[0],cmap='gist_ncar')
    plt.savefig(imageFolder+'specgram/'+title+'.png',bbox_inches='tight', pad_inches=0)

# %%
FPimages = Parallel(n_jobs=60)(delayed(MakeSpectrogramImages)(Moves[i].T, 'Move'+str(i).zfill(4), 300, 512, 505) for i in range(len(Moves)))


