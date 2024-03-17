import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import time
import functools
import pickle
    

from distBox import distBox

import sys
from mpi4py import MPI
from os import environ 
import os
from typing import *                                                                       
#MPIComm = Union[MPI.Intracomm, MPI.Intercomm]
mpi_comm = MPI.COMM_WORLD
myid = mpi_comm.Get_rank()                                                         
mpi_size = mpi_comm.Get_size()        
nprocs=mpi_size

if myid==0:
    print("started")

# set Constants
AirCut = False
RailShape = True
Flaw = True
Absorbing = False

#Dimmesnsion of simulation space in meters
length1 = 20
width1 = 0.1524 # 0.1524
height1 = 0.1524

#Image Folder
imFolder = '/sciclone/scr10/dchendrickson01/EFIT/'
runName = '20m top hit with Flaw at 15x'

#is the rail supported by 0, 1 or 2 ties
Ties = 1

FlawType = 2
# 0 for none, 1 for notch, 2 for crack in head

#Choose ferquency to be used for excitment
frequency = 74574  #brute forced this number to be where simulation frequency 
#            49720  is 2,000,000, alowing for %10 to equal laser 200k same as actual
#            74574  is 3,000,000 hz running, and sample rate %15 is 200k same as actual, if we need more dense
#            99440  is 4,000,000 hz, sample rate of %20 = 200khz
Signalfrequency = 16300

SaveSize = 100  #100 for 15, 150 for 10x: experimentally found for where we don't run out of memory.

cycles = 60

figDPI = 600

#Forcing Function Location and type
# 1 for dropped wheel on top toward start
# 2 for rubbing flange on side
# 3 for plane wave from end
# 4 is square patch rail 20% down
# 5 is a dropped wheel with bounce
# 6 for two rubbing flanges, one on each side
# 7 for two rubbing flanges, on the same side

FFunction = 1

Wheel1Distance = 15.2 # wheel starts X meter down track
# use 15.2 for 20m to maximize good reading time
WheelLoad = 173000 #crane force in Neutons

#MATERIAL 1 ((steel))
pRatio1 = 0.29                                    #poission's ratio in 
yModulus1 = 200 * (10**9)                           #youngs modulus in pascals
rho1 = 7800                                        #density in kg/m^3

DistFromEnd = 0.03

#CALCULATED PARAMETERS FROM INPUTS

mu1 = yModulus1/(2*(1+pRatio1))                    #second Lame Parameter
lmbda1 = 2 * mu1 * pRatio1 / (1 - 2 * pRatio1)     #first Lame Parameter
#Calculate speed of longitudinal and transverse waves in material 1
cl1 = np.sqrt((lmbda1 + 2* mu1)/rho1)
ct1 = np.sqrt(mu1/rho1)

#calculate wave lengths for material 1
omegaL1 = cl1 / frequency
omegaT1 = ct1 / frequency

#Image Folder
imFolder += 'TopHitFlawRun/'

if myid==0:
    if os.path.isdir(imFolder):
        pass
    else:
        os.makedirs(imFolder)


#Set time step and grid step to be 10 steps per frequency and ten steps per wavelength respectively
#ts = 1 / frequency / 10    #time step
gs = (min(omegaL1, omegaT1) /12)    #grid step, omegaL2,omegaT2
ts = gs/((max(cl1,ct1))*(np.sqrt(3)))*0.95 #time step, cl2,ct2


ReadPoint = DistFromEnd #lasers are 5cm from end
FalseWaveTravelDistance = Wheel1Distance + length1 - ReadPoint
FalseWaveTravelTime = FalseWaveTravelDistance / cl1
StepsTillHit = int(FalseWaveTravelTime / ts) + 2 #if plane wave, some wiggle room since also needs to move left/right and up/down

FirstWheelTransverseWaveDistance = length1 - Wheel1Distance + ReadPoint
FirstWheelTransverseFirstReflectionTime = FirstWheelTransverseWaveDistance / ct1

GoodDataPints = (FalseWaveTravelTime - FirstWheelTransverseFirstReflectionTime) / ts

#Run for 3 seconds, what the laser can store:
#runtime = cycles / frequency #cycles / frequency 
#Tsteps = int(math.ceil(runtime / ts)) + 1       #total Time Steps

Tsteps = StepsTillHit + 750  #calculated spereately for needed space to get reflections
#Tsteps = int(3.1*SaveSize)
#Tsteps = 12000


runtime = Tsteps * ts


#number of grid points
gl1 = int(math.ceil(length1 / gs)) +1       #length 
gw1 = int(math.ceil(width1 / gs)) +1       #width
gh1 = int(math.ceil(height1 / gs)) +1       #height

if (myid == 0) :
    print('runtime, gs, ts, fr, gl, gw, gh, Tsteps: ', runtime, gs, ts, 1/ts, gl1, gw1, gh1, Tsteps)

# Keep these as the global values
xmax=gl1-1
ymax=gw1-1
zmax=gh1-1

#####


#MPI EJW Section 1
#extend the length of the beam so that the number of nodes in the x dimmension 
#is the evenly divisible by the number of processors
if (gl1 % nprocs) != 0:
    gl1 += nprocs - (gl1 % nprocs)

#check you did it right
if (gl1 % nprocs) != 0:
    if myid == 0:
        print("Hey, gl1 not divisible by nproc",gl1,nprocs)
        sys.exit()
npx=int(gl1/nprocs)


if myid == 0:
    print("gl1,npx,nproc",gl1,npx,nprocs)

#print(runtime, ts, gs, Tsteps, gl, gh)

if myid == 0:
    print('runtime (s), time step size (s), total # of time steps:', runtime, ts, Tsteps)
    print('grid step size, # of length pts, # of height pts, # of width pts, gl1 loc pts:', gs,gl1,gw1,gh1,npx)

#tensor to store material properties for each point
#0 index is density
#1 index is first Lame Parmaeter
#2 index is second lame parameter

#MPI EJW Section 2 changes
matDensityAll=np.zeros((gl1,gw1,gh1))
matLambdaAll=np.zeros((gl1,gw1,gh1))
matMuAll=np.zeros((gl1,gw1,gh1))
matBCall=np.zeros((gl1,gw1,gh1))
signalLocation=np.zeros((gl1,gw1,gh1))

matDensityAll[:,:,:]=rho1
matLambdaAll[:,:,:]=lmbda1
matMuAll[:,:,:]=mu1
matBCall[:,:,:]=0

## for latter rail section, define the dimmmensions in terms of grid
HeadThickness = 0.05
WebThickness = 0.035
FootThickness = 0.03
HeadWidth = 0.102

relHeadThick = HeadThickness / height1
relWeb = WebThickness / width1
relFoot = FootThickness / height1
relHeadWidth = HeadWidth / width1

relStartHeadThick = 1 - relHeadThick
relStartWeb = 0.5 - (relWeb / 2.0)
relEndWeb = 0.5 + (relWeb / 2.0)
relStartHeadWidth = 0.5 - (relHeadWidth / 2.0)
relEndHeadWidth = 0.5 + (relHeadWidth / 2.0)


gridStartHead = round((gh1-3) * relStartHeadThick) + 1
gridStartWeb = round((gw1-3) * relStartWeb)  + 1
gridEndWeb = round((gw1-3) * relEndWeb)  + 1
gridEndFoot = round((gh1-3) * relFoot)  + 1
gridStartHeadWidth = round((gw1-3) * relStartHeadWidth)  + 1
gridEndHeadWidth = round((gw1-3)  * relEndHeadWidth)  + 1


#Make the Signal Location grid
if FFunction == 1:

    Wheel1Start = int(Wheel1Distance / gs)
    
    sigma = 5 #5 node Stdnard devation
    mu = 6 * sigma # Wheel1Start #center point
    
    bins=np.linspace(0, 12 * sigma, 12 * sigma+1)    
    
    NormDist = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )
    
    temp = np.ones((gridEndHeadWidth - gridStartHeadWidth,len(bins)))
    temp = NormDist * temp

    temp2 = np.ones((5,gridEndHeadWidth - gridStartHeadWidth,len(bins)))
    temp2 = temp * temp2

    temp2 = np.moveaxis(temp2,-1,0)
    temp2 = np.moveaxis(temp2,1,-1)
    
    temp2[:,:,3] = 0.750*temp2[:,:,3]
    temp2[:,:,2] = 0.500*temp2[:,:,2]
    temp2[:,:,1] = 0.250*temp2[:,:,1]
    temp2[:,:,0] = 0.125*temp2[:,:,0]
    
    signalLocation[Wheel1Start - (6 * sigma) : Wheel1Start + (6 * sigma)+1,gridStartHeadWidth:gridEndHeadWidth, -5:] = temp2
    
    specificWheelLoad = WheelLoad / np.sum(temp2[:,:,:4])

    del temp2

    del temp

    
elif FFunction == 2:
     
    signalLocation[14:20,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 1

    signalLocation[20,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[13,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[14:20,gridStartHeadWidth+2:gridStartHeadWidth+3,gridStartHead:zmax-2] = 0.5
    
elif FFunction == 3:
    signalLocation[2:4,:,:] = 1


elif FFunction == 4:
    start = 2*int(gh1/5)
    end = 3 * int(gh1/5)
    signalLocation[2:4,start:end,start:end] = 1

elif FFunction == 5:

    Wheel1Start = int(Wheel1Distance / gs)
    
    sigma = 5 #5 node Stdnard devation
    mu = 6 * sigma # Wheel1Start #center point
    
    bins=np.linspace(0, 12 * sigma, 12 * sigma+1)    
    
    NormDist = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) )
    
    temp = np.ones((gridEndHeadWidth - gridStartHeadWidth,len(bins)))
    temp = NormDist * temp

    temp2 = np.ones((3,gridEndHeadWidth - gridStartHeadWidth,len(bins)))
    temp2 = temp * temp2

    temp2 = np.moveaxis(temp2,-1,0)
    temp2 = np.moveaxis(temp2,1,-1)
    
    signalLocation[Wheel1Start - (6 * sigma) : Wheel1Start + (6 * sigma)+1,gridStartHeadWidth:gridEndHeadWidth, -3:] = temp2
    
    del temp2
    
    specificWheelLoad = WheelLoad / np.sum(temp)

    del temp


elif FFunction == 6:
    
    Wheel1Start = int(Wheel1Distance / gs)
    
    signalLocation[Wheel1Start:Wheel1Start+6,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 1

    signalLocation[Wheel1Start+6,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start-1,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start:Wheel1Start+6,gridStartHeadWidth+2:gridStartHeadWidth+3,gridStartHead:zmax-2] = 0.5

    sep = int(1.360/gs) # Wheel 2 is centered 1.36 meters from wheel 1
    
    signalLocation[Wheel1Start+sep:Wheel1Start+6+sep,gridEndHeadWidth-2:gridEndHeadWidth,gridStartHead:zmax-2] = 1

    signalLocation[Wheel1Start+6+sep,gridEndHeadWidth-2:gridEndHeadWidth,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start-1+sep,gridEndHeadWidth-2:gridEndHeadWidth,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start+sep:Wheel1Start+6+sep,gridEndHeadWidth-3:gridEndHeadWidth-2,gridStartHead:zmax-2] = 0.5
    
    
elif FFunction == 7:
    
    Wheel1Start = int(Wheel1Distance / gs)
    
    signalLocation[Wheel1Start:Wheel1Start+6,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 1

    signalLocation[Wheel1Start+6,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start-1,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start:Wheel1Start+6,gridStartHeadWidth+2:gridStartHeadWidth+3,gridStartHead:zmax-2] = 0.5

    sep = int(1.360/gs) # Wheel 2 is centered 1.36 meters from wheel 1
    
    signalLocation[Wheel1Start+sep:Wheel1Start+6+sep,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 1

    signalLocation[Wheel1Start+6+sep,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start-1+sep,gridStartHeadWidth:gridStartHeadWidth+2,gridStartHead:zmax-2] = 0.5
    signalLocation[Wheel1Start+sep:Wheel1Start+6+sep,gridStartHeadWidth+2:gridStartHeadWidth+3,gridStartHead:zmax-2] = 0.5
    
    

#########
# FUnctions
def JBSU(x,y,z):
    try:
        if (matBCs[x,y,z] == 2 or matBCs[x-1,y,z] == 2 or matBCs[x,y-1,z] == 2 or matBCs[x,y,z-1] == 2):
            pass
        else:
            norm1=(1/gs)*(matLambda[x,y,z]+2*matMu[x,y,z])
            norm2=(1/gs)*(matLambda[x,y,z])

            ds=norm1*(vx[x,y,z]-vx[x-1,y,z])+norm2*(vy[x,y,z]-vy[x,y-1,z]+vz[x,y,z]-vz[x,y,z-1])
            sxx[x,y,z]=sxx[x,y,z]+ds*ts

            ds=norm1*(vy[x,y,z]-vy[x,y-1,z])+norm2*(vx[x,y,z]-vx[x-1,y,z]+vz[x,y,z]-vz[x,y,z-1])
            syy[x,y,z]=syy[x,y,z]+ds*ts

            ds=norm1*(vz[x,y,z]-vz[x,y,z-1])+norm2*(vx[x,y,z]-vx[x-1,y,z]+vy[x,y,z]-vy[x,y-1,z])
            szz[x,y,z]=szz[x,y,z]+ds*ts

        if (matBCs[x,y,z] == 2 or matBCs[x+1,y,z] == 2 or matBCs[x-1,y,z] == 2 or matBCs[x,y+1,z] == 2 
            or matBCs[x,y-1,z] == 2 or matBCs[x,y,z-1] == 2 or matBCs[x+1,y+1,z] == 2):
            pass
        else:
            shearDenomxy=(1/matMu[x,y,z])+(1/matMu[x+1,y,z])+(1/matMu[x,y+1,z])+(1/matMu[x+1,y+1,z])
            shearxy=4*(1/gs)*(1/shearDenomxy)
            ds=shearxy*(vx[x,y+1,z]-vx[x,y,z]+vy[x+1,y,z]-vy[x,y,z])
            sxy[x,y,z]=sxy[x,y,z]+ds*ts

        if (matBCs[x,y,z] == 2 or matBCs[x+1,y,z] == 2 or matBCs[x-1,y,z] == 2 or matBCs[x,y,z+1] == 2 
            or matBCs[x,y,z-1] == 2 or matBCs[x,y-1,z] == 2 or matBCs[x+1,y,z+1] == 2):
            pass
        else:
            shearDenomxz=(1/matMu[x,y,z])+(1/matMu[x+1,y,z])+(1/matMu[x,y,z+1])+(1/matMu[x+1,y,z+1])
            shearxz=4*(1/gs)*(1/shearDenomxz)
            ds=shearxz*(vx[x,y,z+1]-vx[x,y,z]+vz[x+1,y,z]-vz[x,y,z])
            sxz[x,y,z]=sxz[x,y,z]+ds*ts   

        if (matBCs[x,y,z] == 2 or matBCs[x,y,z+1] == 2 or matBCs[x,y,z-1] == 2 or matBCs[x,y+1,z] == 2 
            or matBCs[x,y-1,z] == 2 or matBCs[x-1,y,z] == 2 or matBCs[x,y+1,z+1] == 2):
            pass
        else:
            shearDenomyz=(1/matMu[x,y,z])+(1/matMu[x,y+1,z])+(1/matMu[x,y,z+1])+(1/matMu[x,y+1,z+1])
            shearyz=4*(1/gs)*(1/shearDenomyz)
            ds=shearyz*(vy[x,y,z+1]-vy[x,y,z]+vz[x,y+1,z]-vz[x,y,z])
            syz[x,y,z]=syz[x,y,z]+ds*ts
    except:
        print('Unrecognized BC stress', matBCs[x,y,z],x,y,z)

        
# %%

def JBUV(x,y,z):
    
    if matBCs[x,y,z] == 0: 
        dvxConst=2*(1/gs)*(1/(matDensity[x,y,z]+matDensity[x+1,y,z]))
        dv=dvxConst*( sxx[x+1,y,z]-sxx[x,y,z]
                     +sxy[x,y,z]-sxy[x,y-1,z]
                     +sxz[x,y,z]-sxz[x,y,z-1])
        vx[x,y,z]=vx[x,y,z]+dv*ts
    #x at 0
    elif (matBCs[x,y,z] ==2 or matBCs[x,y-1,z]==2 or matBCs[x,y,z-2]==2):
        pass #requires elements out of the direction
    elif matBCs[x+1,y,z] == 2:
        vx[x,y,z] += 2 * ts/gs * 1/(2 * matDensity[x,y,z]) * ((-2)*sxx[x,y,z])

    elif matBCs[x-1,y,z] ==2 :
        vx[x,y,z] += 2 * ts/gs * 1/(2 * matDensity[x,y,z]) * ((2)*sxx[x+1,y,z])

    else:
        dvxConst=2*(1/gs)*(1/(matDensity[x,y,z]+matDensity[x+1,y,z]))
        dv=dvxConst*( sxx[x+1,y,z]-sxx[x,y,z]
                     +sxy[x,y,z]-sxy[x,y-1,z]
                     +sxz[x,y,z]-sxz[x,y,z-1])
        vx[x,y,z]=vx[x,y,z]+dv*ts
    
    #Vy cases
    if matBCs[x,y,z] == 0: 
        dvyConst=2*(1/gs)*(1/(matDensity[x,y,z]+matDensity[x,y+1,z]))
        dv=dvyConst* ( sxy[x,y,z]-sxy[x-1,y,z]
                      +syy[x,y+1,z]-syy[x,y,z]
                      +syz[x,y,z]-syz[x,y,z-1])
        vy[x,y,z]=vy[x,y,z]+dv*ts
    #y = 0
    elif (matBCs[x,y,z] ==2 or matBCs[x-1,y,z] == 2 or matBCs[x,y,z-1] == 2):
        pass  #requires elements out of the direction
    elif matBCs[x,y+1,z] == 2:
        vy[x,y,z] += 2 * ts/gs * 1/(2 * matDensity[x,y,z]) * ((-2)*syy[x,y,z])
    elif matBCs[x,y-1,z] == 2:
        vy[x,y,z] += 2 * ts/gs * 1/(2 * matDensity[x,y,z]) * ((2)*syy[x,y+1,z])
    else:
        dvyConst=2*(1/gs)*(1/(matDensity[x,y,z]+matDensity[x,y+1,z]))
        dv=dvyConst* ( sxy[x,y,z]-sxy[x-1,y,z]
                      +syy[x,y+1,z]-syy[x,y,z]
                      +syz[x,y,z]-syz[x,y,z-1])
        vy[x,y,z]=vy[x,y,z]+dv*ts

    #Vz cases
    if matBCs[x,y,z] ==0:
        dvzConst=2*(1/gs)*(1/(matDensity[x,y,z]+matDensity[x,y,z+1]))
        dv=dvzConst*( sxz[x,y,z]-sxz[x-1,y,z]
                     +syz[x,y,z]-syz[x,y-1,z]
                     +szz[x,y,z+1]-szz[x,y,z])
        vz[x,y,z]=vz[x,y,z]+dv*ts
    #z at 0
    elif (matBCs[x,y,z] == 2 or matBCs[x-1,y,z] == 2 or matBCs[x,y-1,z]==2):
        pass
    elif matBCs[x,y,z+1] == 2:
        vz[x,y,z] += 2 * ts/gs *(1/(2 * matDensity[x,y,z])) * ((-2)*szz[x,y,z])
    elif matBCs[x,y,z-1] == 2:
        vz[x,y,z] += 2 * ts/gs *(1/(2 * matDensity[x,y,z])) * ((2)*szz[x,y,z+1])
    else:
        dvzConst=2*(1/gs)*(1/(matDensity[x,y,z]+matDensity[x,y,z+1]))
        dv=dvzConst*( sxz[x,y,z]-sxz[x-1,y,z]
                     +syz[x,y,z]-syz[x,y-1,z]
                     +szz[x,y,z+1]-szz[x,y,z])
        vz[x,y,z]=vz[x,y,z]+dv*ts

def setSimSpaceBC99(matBCs):
    
    matBCs[0,:,:]=99
    matBCs[xmax,:,:]=99
    matBCs[:,0,:]=99
    matBCs[:,ymax,:]=99
    matBCs[:,:,0]=99
    matBCs[:,:,zmax]=99
    
    return matBCs

def setSimSpaceBCs(matBCs):
    #Second Dimmension boundaries /y
    matBCs[:,0,:]=2
    matBCs[:,1,:]=1
    matBCs[:,ymax,:]=2
    matBCs[:,ymax-1,:]=2
    matBCs[:,ymax-2,:]=1

    #Third Dimmension Boundaries /z
    matBCs[:,:,0]=2
    matBCs[:,2:ymax-1,1]=1
    matBCs[:,:,zmax]=2
    matBCs[:,:,zmax-1]=2
    matBCs[:,2:ymax-1,zmax-2]=1
    
    #First Dimmension Boundaries /x
    #   handled different if this is going to be calculated by node
    #   others c does it different, but they split between nodes before calculating
    #   here we calculate the whole set and then parse
    matBCs[0,:,:]=2
    matBCs[1,2:ymax-1,1:zmax-1]=1
    matBCs[xmax,:,:]=2
    matBCs[xmax-1,:,:]=2
    matBCs[xmax-2,1:ymax-1,1:zmax-1]=1
    
    return matBCs

def setRailBCs(matBCs):
    #top of foot
    matBCs[:,1:gridStartWeb,gridEndFoot]=1
    matBCs[:,gridEndWeb:ymax-1,gridEndFoot]=1

    #Sides Web
    matBCs[:,gridStartWeb,gridEndFoot:gridStartHead] = 1
    matBCs[:,gridEndWeb,gridEndFoot:gridStartHead] =1

    #bottom Head
    matBCs[:,gridStartHeadWidth:gridStartWeb+1,gridStartHead] = 1
    matBCs[:,gridEndWeb:gridEndHeadWidth,gridStartHead] = 1

    #Sides HEad
    matBCs[:,gridStartHeadWidth,gridStartHead:zmax-1] = 1
    matBCs[:,gridEndHeadWidth,gridStartHead:zmax-1] = 1

    #air beside Web
    matBCs[:,1:gridStartWeb,gridEndFoot+1:gridStartHead] = 2
    matBCs[:,gridEndWeb+1:ymax,gridEndFoot+1:gridStartHead] = 2

    #air beside head
    matBCs[:,1:gridStartHeadWidth,gridStartHead:zmax] = 2
    matBCs[:,gridEndHeadWidth+1:ymax,gridStartHead:zmax] = 2

    
    return matBCs

def MakeFlaw(matBCs, FlawType = 1):
    if FlawType ==1:
        #notch in plate
        MidPoint = int(gl1/2)
        StartTrans = int(gl1/5)*2
        EndTrans = int(gl1/5)*3

        TransToEnd = gl1-EndTrans
        MidTransToEnd = int(TransToEnd/2)+EndTrans
        QuarterTrans = int((EndTrans-StartTrans)/4)

        StartFlawX = MidTransToEnd - QuarterTrans
        EndFlawX = MidTransToEnd + QuarterTrans

        StartFlawY = MidPoint - QuarterTrans
        EndFlawY = MidPoint + QuarterTrans

        VertFlaw = int(gh1/8)
        VertStart = zmax - VertFlaw

        #main hole
        matBCs[StartFlawX:EndFlawX,StartFlawY:EndFlawY,VertStart:] = 2

        #edges
        matBCs[StartFlawX:EndFlawX,StartFlawY:EndFlawY,VertStart-1] = 1
        matBCs[StartFlawX-1,StartFlawY-1:EndFlawY+1,VertStart:zmax-2]=1
        matBCs[EndFlawX+1,StartFlawY-1:EndFlawY+1,VertStart:zmax-2]=1
        matBCs[StartFlawX-1:EndFlawX+1,StartFlawY-1,VertStart:zmax-2]=1
        matBCs[StartFlawX-1:EndFlawX+1,EndFlawY+1,VertStart:zmax-2]=1
        
    elif FlawType ==2:
        #crack in rail head
        #Wheel2StartPoint = int(1.360/gs) + Wheel1Start
        #StartPoint = int(Wheel2StartPoint+((xmax - Wheel2StartPoint)/2))
        
        StartPoint = int(19/gs)
        
        matBCs[StartPoint-1:StartPoint+1,gridStartHeadWidth:gridEndHeadWidth,zmax-int(0.03/gs):zmax-1] = 1
        matBCs[StartPoint,gridStartHeadWidth:gridEndHeadWidth,zmax-int(0.03/gs):zmax-1] = 2
        
    return matBCs
    

#matBCs = setSimSpaceBC99(matBCs)
matBCall = setSimSpaceBCs(matBCall)
    
if RailShape:
    #matDensity,matLambda,matMu = setAirCut(matDensity,matLambda,matMu)
    matBCall = setRailBCs(matBCall)
    #matBCs = addTies(matBCs,Ties)

#Add Flaw
if Flaw:
    matBCall = MakeFlaw(matBCall, FlawType)

#Create End Damping to make asorbing boundary
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

Absorber = np.ones((gl1,gw1,gh1))
StepAbsorption = 0.5
AbsorptionRange = 101
if Absorbing:
    for x in range(AbsorptionRange):
        Absorber[x,:,:] = sigmoid((x-(int(AbsorptionRange)/2))/int(AbsorptionRange/10))

    
#If there are ties:
if Ties > 0:
    TieWidth = .2
    if Ties == 1:
        TieSpacing =  .54
    elif Ties == 2:
        TieSpacing =  .74
    else:
        TieSpacing =  .34
    
    TieGridWidth = int(TieWidth / gs)
    TieGridSpacing = int(TieSpacing / gs)
    
    i = 0
    c = 0
    TieSpace = 0
    while i < gl1:
        if TieSpace == 0:
            c+=1
            if c == 1 and i != 0:
                Absorber[i-1,:,3] = 0.8
                Absorber[i-1,:,2] = 0.6
                Absorber[i-1,:,1] = 0.4
                Absorber[i-2,:,2] = 0.8
                Absorber[i-2,:,1] = 0.6
                Absorber[i-3,:,1] = 0.8
            elif c == TieGridWidth:
                Absorber[i+1,:,3] = 0.8
                Absorber[i+1,:,2] = 0.6
                Absorber[i+1,:,1] = 0.4
                Absorber[i+2,:,2] = 0.8
                Absorber[i+2,:,1] = 0.6
                Absorber[i+3,:,1] = 0.8
                c=0
                TieSpace = 1
                
            Absorber[i,:,1] = 0.2
            Absorber[i,:,2] = 0.4
            Absorber[i,:,3] = 0.6
            Absorber[i,:,4] = 0.8
            
            if TieSpace == 0 and c == TieGridSpacing:
                c = 0
                TimeSpace = 1
                
        i+=1
            
    
#define sine-exponential wave excitation

timeVec=np.linspace(0,runtime,Tsteps)

#MPI EJW Section #3 changes
#radius
r=3
inputx=2
inputy=int(gw1/2)
inputz=int(gh1/2)

# get loc by formula

inputid=int(inputx / npx)
inputlocx=int(inputx - inputid*npx+1)

RecordPlane = xmax - int(DistFromEnd / gs)
RecordNode = int(RecordPlane/npx)
InNodeRecordPlane = int(RecordPlane - RecordNode*npx+1)

szzConst=2*ts/(gs*rho1)

amp=100000
if FFunction == 1:
    decayRate = 40000
elif FFunction ==5:
    decayRate = 50000
else:
    decayRate= 0
sinConst=ts*amp/rho1

sinInputSignal=sinConst*np.sin(2*np.pi*Signalfrequency*timeVec)*np.exp(-decayRate*timeVec)
#sinInputSignal[int(.1*Tsteps+1):] = 0

if FFunction == 5:
    start = int(0.1 * Tsteps)
    end = Tsteps - start
    sinInputSignal[start:] += sinInputSignal[:end]
    if myid == 0:
        fig = plt.figure()
        plt.plot(sinInputSignal)
        plt.savefig(imFolder+'signal.png')
        plt.show()
        plt.close()

# MPI EJW Section #4 changes 

#initialize fields
vx=np.zeros((npx+2,gw1,gh1))
vy=np.zeros((npx+2,gw1,gh1))
vz=np.zeros((npx+2,gw1,gh1))

sxx=np.zeros((npx+2,gw1,gh1))
syy=np.zeros((npx+2,gw1,gh1))
szz=np.zeros((npx+2,gw1,gh1))
sxy=np.zeros((npx+2,gw1,gh1))
sxz=np.zeros((npx+2,gw1,gh1))
syz=np.zeros((npx+2,gw1,gh1))


#record the signal at a specified location
### ADD map function for this
signalLocx=int(gl1/2)
signalLocy=int(gw1/2)
signalLocz=int(gh1/2)

#SAME AS INPUTZ?
signalLocxid=int(signalLocx / npx)
signalLocxlocx=int(signalLocx - myid*npx+1)

vxSignal=np.zeros(Tsteps)
vySignal=np.zeros(Tsteps)
vzSignal=np.zeros(Tsteps)

# Grab splits and offsets for scattering arrays
# Only thing to scatter is matPropsglob
# v's and s's are zero to start + source applied later 
# in single proc's array
if myid == 0:
    split=np.zeros(nprocs)
    split[:]=gw1*gh1*npx

    offset=np.zeros(nprocs)
    for i in range(nprocs):
        offset[i]=i*gw1*gh1*npx
else:
    split=None
    offset=None

split=mpi_comm.bcast(split)
offset=mpi_comm.bcast(offset)

matDensity = np.zeros((npx,gw1,gh1))
matLambda = np.zeros((npx,gw1,gh1))
matMu = np.zeros((npx,gw1,gh1))
matBCs = np.zeros((npx,gw1,gh1))
signalloc = np.zeros((npx,gw1,gh1))
mpiAbsorber=np.zeros((npx,gw1,gh1))

mpi_comm.Scatterv([matDensityAll,split,offset,MPI.DOUBLE], matDensity)
mpi_comm.Scatterv([matLambdaAll,split,offset,MPI.DOUBLE], matLambda)
mpi_comm.Scatterv([matMuAll,split,offset,MPI.DOUBLE], matMu)
mpi_comm.Scatterv([matBCall,split,offset,MPI.DOUBLE], matBCs)
mpi_comm.Scatterv([signalLocation[:,:,:],split,offset,MPI.DOUBLE], signalloc)
mpi_comm.Scatterv([Absorber,split,offset,MPI.DOUBLE], mpiAbsorber)


matDensity=distBox(matDensity,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
matLambda=distBox(matLambda,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
matMu=distBox(matMu,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
matBCs=distBox(matBCs,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
signalloc=distBox(signalloc,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
mpiAbsorber=distBox(mpiAbsorber,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        

#Now slab has local versions with ghosts of matProps
if (myid == 0) :
    
    ## All signals at 4 nodes from end
    FromEnd = int(.05 / gs) # 5 cm from end, about where laser recording is done
    
    # Top of rail
    FSignalLocX= gl1-FromEnd
    FSignalLocY=int(gw1/2)
    FSignalLocZ=gh1-2

    ## End halfway up head
    BSignalLocX=gl1-FromEnd
    BSignalLocY=int(gw1/2)
    BSignalLocZ=gh1-int((gh1-gridStartHead)/2)

    ## left of head
    USignalLocX=gl1-FromEnd
    USignalLocY=gridStartHeadWidth
    USignalLocZ=gh1-int((gh1-gridStartHead)/2)

    ## right of head
    DSignalLocX=gl1-FromEnd
    DSignalLocY=gridEndHeadWidth
    DSignalLocZ=gh1-int((gh1-gridStartHead)/2)

    ## right of web
    RSignalLocX=gl1-FromEnd
    RSignalLocY=gridStartWeb
    RSignalLocZ=int(gh1/2)

    ## Left of Web
    LSignalLocX=gl1-FromEnd
    LSignalLocY=gridEndWeb
    LSignalLocZ=int(gh1/2)

    ## End of Web
    MSignalLocX=gl1-2
    MSignalLocY=int(gw1/2)
    MSignalLocZ=int(gh1/2)


    #signal locations going to be a quarter of the way in the middle from the 
    # Front, Back, Up side, Down side, Right, Left, and Middle Middle Middle
    FSignal=np.zeros((Tsteps,3))
    BSignal=np.zeros((Tsteps,3))
    USignal=np.zeros((Tsteps,3))
    DSignal=np.zeros((Tsteps,3))
    RSignal=np.zeros((Tsteps,3))
    LSignal=np.zeros((Tsteps,3))
    MSignal=np.zeros((Tsteps,3))
    
    # switch to 1000 from Tsteps for size, and then added a saving every 1000 to fit in memory
    Movements = np.zeros((gl1,gw1,gh1,SaveSize))
    MovementsX = np.zeros((gl1,gw1,gh1,SaveSize))
    MovementsY = np.zeros((gl1,gw1,gh1,SaveSize))
    MovementsZ = np.zeros((gl1,gw1,gh1,SaveSize))
    DisX = np.zeros((gl1,gw1,gh1))
    DisY = np.zeros((gl1,gw1,gh1))
    DisZ = np.zeros((gl1,gw1,gh1))
    
    
    Parameters = {"AirCut" : AirCut,
                  "RailShape":  RailShape,
                  "Flaw" : Flaw,
                  "AbsorptionOn" : Absorbing,
                  "Length" : length1,
                  "Width" : width1,
                  "Height" : height1,
                  "SaveFolder" : imFolder,
                  "RunTitle" : runName,
                  "TiesIncluded" : Ties,
                  "GridDesignFrequency" : frequency,
                  "InputSignalFrequency" : Signalfrequency,
                  "SimulationCycleLength" : cycles,
                  "ForcingFuctionNumber" : FFunction,
                  "PerWheelForce" : WheelLoad,
                  "Wheel1Start" : Wheel1Distance,
                  "PoisonsRatio" : pRatio1,
                  "YoungsModulous" : yModulus1,
                  "MaterialDensity" : rho1,
                  "LongitudinalWaveSpeed" : cl1,
                  "TransverseWaveSpeeed" : ct1,
                  "TimeStep" : ts,
                  "GridStep" : gs,
                  "RunTime" : runtime,
                  "TimeStepsSimLength" : Tsteps,
                  "GridLengthNodes" : gl1,
                  "GridWidthNodes" : gw1,
                  "GridHeightNodes" : gh1,
                  "LargestXnode" : xmax,
                  "LargestYnode" : ymax,
                  "LargestZnode" : zmax,
                  "SaveEveryXStep" : SaveSize,
                  "HeightStartHeadNode" : gridStartHead,
                  "WidthStartWebNode" : gridStartWeb,
                  "WidthEndWebNode" : gridEndWeb,
                  "HeightEndFootNode" : gridEndFoot,
                  "WidthStartHeadNode" : gridStartHeadWidth,
                  "WidthEndHeadNode" : gridEndHeadWidth,
                  "AbsorberLengthNodes" : AbsorptionRange,
                  "AbsorptionPerNode" : StepAbsorption,
                  "ExpectedGoodData" : GoodDataPints,
                  "PoisonsRatio" : pRatio1,
                  "FlawType" : FlawType,
                  "YoungsModulous" : yModulus1,
                  "DecayRate" : decayRate,
                  "Recording plane node" : RecordPlane,
                  "Recording plane m" : DistFromEnd
    }
                  
    file=open(imFolder+'Parameters.p','wb')
    pickle.dump(Parameters,file)
    file.close()
    
    print(Parameters)
    
    del Parameters
    
    MinMax = np.zeros((4,2))
    
    j=0

stime = time.time()

DisX = np.zeros((gl1,gw1,gh1))
DisY = np.zeros((gl1,gw1,gh1))
DisZ = np.zeros((gl1,gw1,gh1))

if myid == RecordNode:
    Records = np.zeros((gh1,gw1,Tsteps))
    
for t in range(0,Tsteps):
    if FFunction == 1:
        vz -= signalloc * specificWheelLoad / rho1 * ts
    elif FFunction ==2:
        vz += signalloc * sinInputSignal[t]
    elif FFunction ==3:
        vx += signalloc * sinInputSignal[t]
    elif FFunction ==4:
        vx += signalloc * sinInputSignal[t]
    elif FFunction ==5:
        vz -= signalloc * sinInputSignal[t]
    elif FFunction ==6:
        vz -= signalloc * sinInputSignal[t]
    elif FFunction ==7:
        vx -= signalloc * sinInputSignal[t]
        vz -= signalloc * sinInputSignal[t]

    for x in range(1,npx+1):
        for y in range(0,ymax):
            for z in range(0,zmax):
                JBSU(x,y,z)


    # cut boundaries off of arrays
    sxxt=sxx[1:npx+1,:,:]
    syyt=syy[1:npx+1,:,:]
    szzt=szz[1:npx+1,:,:]
    sxyt=sxy[1:npx+1,:,:]
    sxzt=sxz[1:npx+1,:,:]
    syzt=syz[1:npx+1,:,:]

    # redistrubute ghost/boundary values
    sxx=distBox(sxxt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    syy=distBox(syyt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    szz=distBox(szzt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    sxy=distBox(sxyt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    sxz=distBox(sxzt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    syz=distBox(syzt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        

    #if the forcing function is a stress

    for x in range(1,npx+1):
        for y in range(0,ymax):
            for z in range(0,zmax):
                JBUV(x,y,z)
    vx*=mpiAbsorber
    vy*=mpiAbsorber
    vz*=mpiAbsorber
        
    # cut boundaries off of arrays
    vxt=vx[1:npx+1,:,:]
    vyt=vy[1:npx+1,:,:]
    vzt=vz[1:npx+1,:,:]

    # redistrubute ghost/boundary values
    vx=distBox(vxt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    vy=distBox(vyt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        
    vz=distBox(vzt,myid,gl1,gw1,gh1,npx,nprocs,mpi_comm)        

    #record signals
    if (myid==signalLocxid) :
        vxSignal[t]=vx[signalLocxlocx,signalLocy,signalLocz]
        vySignal[t]=vy[signalLocxlocx,signalLocy,signalLocz]
        vzSignal[t]=vz[signalLocxlocx,signalLocy,signalLocz]

    # save vx cut figure
    # ADD GATHER for plotting

    
    vxg = np.zeros((gl1,gw1,gh1))
    vxt=vx[1:npx+1,:,:]        
    mpi_comm.Gatherv(vxt,[vxg,split,offset,MPI.DOUBLE])

    vzg = np.zeros((gl1,gw1,gh1))
    vzt=vz[1:npx+1,:,:]        
    mpi_comm.Gatherv(vzt,[vzg,split,offset,MPI.DOUBLE])

    vyg = np.zeros((gl1,gw1,gh1))
    vyt=vy[1:npx+1,:,:]        
    mpi_comm.Gatherv(vyt,[vyg,split,offset,MPI.DOUBLE])

    if t%SaveSize == 0:
        sxxg = np.zeros((gl1,gw1,gh1))
        sxxt=vx[1:npx+1,:,:]        
        mpi_comm.Gatherv(sxxt,[sxxg,split,offset,MPI.DOUBLE])

        syyg = np.zeros((gl1,gw1,gh1))
        syyt=vz[1:npx+1,:,:]        
        mpi_comm.Gatherv(syyt,[syyg,split,offset,MPI.DOUBLE])

        szzg = np.zeros((gl1,gw1,gh1))
        szzt=vy[1:npx+1,:,:]        
        mpi_comm.Gatherv(szzt,[szzg,split,offset,MPI.DOUBLE])

        sxyg = np.zeros((gl1,gw1,gh1))
        sxyt=vx[1:npx+1,:,:]        
        mpi_comm.Gatherv(sxyt,[sxyg,split,offset,MPI.DOUBLE])

        syzg = np.zeros((gl1,gw1,gh1))
        syzt=vz[1:npx+1,:,:]        
        mpi_comm.Gatherv(syzt,[syzg,split,offset,MPI.DOUBLE])

        sxzg = np.zeros((gl1,gw1,gh1))
        sxzt=vy[1:npx+1,:,:]        
        mpi_comm.Gatherv(sxzt,[sxzg,split,offset,MPI.DOUBLE])
    
    if myid == RecordNode:
        Records[:,:,t] = vx[InNodeRecordPlane,:,:]
        if t%SaveSize == 0:
            file=open(imFolder+'SavePlane.p','wb')
            pickle.dump(Records,file)
            file.close()
    

    if myid==0:
        
        DisX += vxg[:,:,:] * ts
        DisY += vyg[:,:,:] * ts
        DisZ += vzg[:,:,:] * ts
        Movements[:,:,:,t%SaveSize] = np.sqrt(DisX**2 + DisY**2 + DisZ**2)  # ad the modulous 1000 for big model and multi save
        MovementsX[:,:,:,t%SaveSize] = DisX
        MovementsY[:,:,:,t%SaveSize] = DisY
        MovementsZ[:,:,:,t%SaveSize] = DisZ
            
        if t%SaveSize == 0:
            file=open(imFolder+'MovementsR2MM'+str(j).zfill(3)+'.p','wb')
            pickle.dump([Movements, MovementsX, MovementsY, MovementsZ],file)
            file.close()

            file=open(imFolder+'CurrentState.p','wb')
            pickle.dump([t,vxg,vyg,vzg,
                        sxxg,syyg,szzg,
                        sxyg,syzg,sxzg],file)
            file.close()
            
            j+=1

    # Collect vx, sxx checksum contributions for printing
    vxt=vx[1:npx+1,:,:]
    sxxt=sxx[1:npx+1,:,:]

    ckvs=np.array(0.0,'d')
    ckss=np.array(0.0,'d')
    
    ckv=np.sum(np.absolute(vxt))
    cks=np.sum(np.absolute(sxxt))
    mpi_comm.Reduce(ckv,ckvs,op=MPI.SUM,root=0)
    mpi_comm.Reduce(cks,ckss,op=MPI.SUM,root=0)

    if (myid == 0 ):
        print(t,'/',Tsteps-1,'checksums vx, sxx:',ckvs,ckss, int((time.time()-stime)/60.0*100)/100)
    sys.stdout.flush()


if myid ==0:
    #print(MidMatrix)
       
    file=open(imFolder+'MovementsR2MM'+str(j).zfill(3)+'.p','wb')
    pickle.dump([Movements[:,:,:,:Tsteps%SaveSize], MovementsX[:,:,:,:Tsteps%SaveSize], 
                 MovementsY[:,:,:,:Tsteps%SaveSize], MovementsZ[:,:,:,:Tsteps%SaveSize]],file) 
    #don't save the trailing bad data from the last
    file.close()
    
    file=open(imFolder+'MinMax.p','wb')
    pickle.dump(MinMax, file) 
    file.close()

if myid == RecordNode:
    file=open(imFolder+'SavePlane.p','wb')
    pickle.dump(Records,file)
    file.close()
    
    
