#Standard Header used on the projects

#first the major packages used for math and graphing
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scipy.special as sp
import pandas as pd


#Extra Headers:fp
import os as os
import pywt as py
import statistics as st
import os as os
import random
from joblib import Parallel, delayed
from pywt._extensions._pywt import (DiscreteContinuousWavelet, ContinuousWavelet,
                                Wavelet, _check_dtype)
from pywt._functions import integrate_wavelet, scale2frequency
from time import time as ti


Header = np.array(['Date', 'Hour', 'Minute', 'Second', 'Sec Fraction', 'Sen0x', 'Sen0y', 'Sen0z', 'Sen1x', 'Sen1y', 'Sen1z', 'Sen2x', 'Sen2y', 'Sen2z', 'Sen3x', 'Sen3y', 'Sen3z', 'Sen4x', 'Sen4y', 'Sen4z', 'Sen5x', 'Sen5y', 'Sen5z'])

Directory = 'C:\\Users\\Dan\\Desktop\\Temp\\'
files = os.listdir(Directory)

start = 0
end = 16000000

size = 5000
Arange = 50
coord = 2

Saving = True
location = Directory
Titles = False

for Filename in files:
    if Filename[-4:] ==  '.csv':
        ODataSet = np.genfromtxt(open(Directory+'/'+Filename,'r'), delimiter=',',skip_header=1)
        length = np.shape(ODataSet)[0]

        results =[]
        Results = []
        for i in range(2):
            coord = i+5
            for j in range(np.shape(ODataSet)[0]-1):
                results.append(np.sign(ODataSet[j+1,coord]*ODataSet[j,coord])*np.abs(np.abs(ODataSet[j+1,coord])-np.abs(ODataSet[j,coord])))

            print(np.average(results), st.stdev(results))
