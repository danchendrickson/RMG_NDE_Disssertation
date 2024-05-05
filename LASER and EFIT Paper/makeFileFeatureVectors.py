#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.special as sp
import platform

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

#Standard cycle for collors and line styles
default_cycler = (cycler('color', ['0.00', '0.40', '0.60', '0.70']) + cycler(linestyle=['-', '--', ':', '-.']))
plt.rc('axes', prop_cycle=default_cycler)

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
    folder = '/sciclone/scr10/dchendrickson01/Laser Data/'
    SaveFolder='/sciclone/scr10/dchendrickson01/LaserFPVectorsdb3/'
elif Computer == "Desktop":
    rootfolder = location
    folder = rootfolder + "Recordings2\\SubSet\\"
elif Computer =="WinLap":
    rootfolder = location
    folder = rootfolder + "Recordings2\\"   
elif Computer == "LinLap":
    rootfolder = '/home/dan/Data/'
    folder = rootfolder + 'Recordings2/'
elif Computer =='PortLap':
    rootfolder = location 
    folder = rootfolder + 'Recordings2\\'
    
#Extra Headers:
import os as os
import statistics as st
import random
import multiprocessing
from joblib import Parallel, delayed
import time
import CoreFunctions as cf
import DWFT as df
import pywt
import scipy.signal as signal
from matplotlib import ticker

procs = multiprocessing.cpu_count()-1

my_cmap = plt.get_cmap('gray')

files = os.listdir(folder)

wvlt = 'db3'

f2 = []
for file in files:
    #if file[:8] == 'stack 27' or file[:8] =='Stack 27':
    if os.path.isdir(SaveFolder+file):
        #f2.append(file)
        pass
    else:
        f2.append(file)
        #pass
        
def focusArea(file, start=0, end=600000, graph=True):
    #file = f2[f]
    ODataSet = np.genfromtxt(open(folder+'/'+file,'r'), delimiter=',',skip_header=1)
    ODataSet = ODataSet[:,2:5]
    for coord in range(2):
        for j in range(np.shape(ODataSet)[0]-1):
            try:
                ODataSet[j,coord] = float(ODataSet[j,coord])
            except:
                ODataSet[j,coord] = ODataSet[j-1,coord]
            if ODataSet[j,coord] == -999.999:
                ODataSet[j,coord] = ODataSet[j-1,coord]

    Diffs = np.zeros(np.shape(ODataSet)[0])
    for j in range(np.shape(ODataSet)[0]-1):
        Diffs[j] = ODataSet[j,0] - ODataSet[j,1]
    norm = np.average(ODataSet[:,0])
    ODataSet[:,0]-=norm
    norm = np.average(ODataSet[:,1])
    ODataSet[:,1]-=norm
    norm = np.average(Diffs)
    Diffs[:]-=norm
    Cdiff = cf.Smoothing(Diffs,2,dets_to_remove=3)
    
    if end > len(Cdiff):
        end = len(Cdiff)
    
    if graph:
        fig,axs = plt.subplots(2,figsize=(6,4), dpi=600)
        plt.subplots_adjust(hspace=0.5)
        plt.title(str(f)+' ' + file)
        axs[0].plot(Cdiff[25:-50], linewidth=0.5)
        axs[1].plot(np.linspace(start,end,end-start),Cdiff[start:end]-np.average(Cdiff[start:end]), linewidth=0.5)
        plt.show()
    
    return Cdiff

def makeFV(file):
    FullMatrix = np.zeros((1,29))
    #try:
    cDiff=focusArea(file,graph=False)

    fpCount = 0

    fpRange = 1000
    Loops = int(cDiff.shape[0]/fpRange)

    for i in range(Loops):
        Vect=[]
        test, fbcom = df.getProcessedFP(cDiff[i*fpRange:(i+1)*fpRange],'morl')

        for k in range(fbcom.shape[0]-1):
            j = k+1
            Vect.append(df.FPFeatureVector(test,j)[0])

        Vect=np.matrix(Vect) 
        fbcom=np.matrix(fbcom)
        fbcom[:,0]+=i*fpRange
        fbcom=np.concatenate((fbcom,np.matrix(np.arange(fpCount,fpCount+fbcom.shape[0])).T),axis=1)
        #fbcom=fbcom.reshape(fbcom.shape[0],fbcom.shape[2])

        fpCount+=fbcom.shape[0]

        CombiData = np.concatenate((fbcom,Vect),axis=1)

        FullMatrix=np.concatenate((FullMatrix,CombiData),axis=0)    

    np.savetxt(SaveFolder+file,FullMatrix,delimiter=",",fmt='%s')
    #except:
    #    pass
    print(file,FullMatrix.shape)
    
    return 1
        
result = Parallel(n_jobs=31)(delayed(makeFV)(file) for file in f2)

#h = makeFV(f2[3])