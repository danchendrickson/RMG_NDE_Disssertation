
dataSize = 'big'

import math
import numpy as np
import scipy.stats as st
import scipy.signal as ss
import pywt
from pywt._extensions._pywt import (DiscreteContinuousWavelet, ContinuousWavelet,
                                Wavelet, _check_dtype)
from pywt._functions import integrate_wavelet, scale2frequency 
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from cycler import cycler
import platform
import datetime
from itertools import compress

import os
#import cv2

import multiprocessing
from joblib import Parallel, delayed

#If being run by a seperate file, use the seperate file's graph format and saving paramaeters
#otherwise set what is needed
if not 'Saving' in locals():
    Saving = True
if not 'Titles' in locals():
    Titles = True
if not 'Ledgends' in locals():
    Ledgends = True
if not 'FFormat' in locals():
    FFormat = '.png'
if not 'BWImage' in locals():
    UseWBstyle = False

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
    if dataSize == 'big':
        folder = '/sciclone/scr10/dchendrickson01/CraneData/'
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
elif Computer =='PortLap':
    rootfolder = location 
    folder = rootfolder + 'SmallCopy\\'

#Standard cycle for collors and line styles
if UseWBstyle:
    default_cycler = (cycler('color', ['0.00', '0.40', '0.60', '0.70']) + cycler(linestyle=['-', '--', ':', '-.']))
    plt.rc('axes', prop_cycle=default_cycler)
    my_cmap = plt.get_cmap('gray')

    PlotWidthIn = 11
    PlotHeightIn = 3.75
    PlotDPI = 3000

beta_a = 2
beta_b = 5
beta_cycles = 4
beta_sineCosine = 1
WaveletToUse = 'beta'
#scales = np.linspace(0,2000,1001, dtype=int)
scales = 500
spacer = 10

num_cores = multiprocessing.cpu_count() -1

#SensorPositonFile = rootfolder + 'SensorStatsSmall.csv'
#OutputVectors = np.genfromtxt(open(SensorPositonFile,'r'), delimiter=',',skip_header=1,dtype=int, missing_values=0)
#OutputTitles = OutputVectors[0,:]
#OutputVectors = OutputVectors[1:,:]


def BetaWavelet(sizes, a = beta_a, b = beta_b, sineCycle = beta_cycles, cosineCycle = 0):
    beta = np.zeros(sizes)
    beWave = np.zeros(sizes)
    x = np.zeros(sizes)
    for i in range(sizes):
        j = i / sizes
        beta[i] = st.beta.pdf(j,a,b)
        if sineCycle == 0:
            beWave[i] = beta[i] * math.cos(j*math.pi*cosineCycle)
        else:
            beWave[i] = beta[i] * math.sin(j*math.pi*sineCycle) * math.cos(j*math.pi*cosineCycle)
        x[i]=j
        
    #beWav2 = beWave[::-1]
    return beWave, x

def cwt_fixed(data, scales, wavelet, scalespace =1, sampling_period=1., betaParameters = [10000, beta_a,  beta_b, beta_cycles, 0]):
    """
    COPIED AND FIXED FROM pywt.cwt TO BE ABLE TO USE WAVELET FAMILIES SUCH
    AS COIF AND DB

    COPIED From Spenser Kirn
    
    All wavelet work except bior family, rbio family, haar, and db1.
    
    cwt(data, scales, wavelet)

    One dimensional Continuous Wavelet Transform.

    Parameters
    ----------
    data : array_like
        Input signal
    scales : array_like
        scales to use
    wavelet : Wavelet object or name
        Wavelet to use
    sampling_period : float
        Sampling period for frequencies output (optional)

    Returns
    -------
    coefs : array_like
        Continous wavelet transform of the input signal for the given scales
        and wavelet
    frequencies : array_like
        if the unit of sampling period are seconds and given, than frequencies
        are in hertz. Otherwise Sampling period of 1 is assumed.

    Notes
    -----
    Size of coefficients arrays depends on the length of the input array and
    the length of given scales.

    Examples
    --------
    >>> import pywt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(512)
    >>> y = np.sin(2*np.pi*x/32)
    >>> coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
    >>> plt.matshow(coef) # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP
    ----------
    >>> import pywt
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> t = np.linspace(-1, 1, 200, endpoint=False)
    >>> sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
    >>> widths = np.arange(1, 31)
    >>> cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
    >>> plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    ...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
    >>> plt.show() # doctest: +SKIP
    """
    
    #scales = get_primelist(10000)
    
    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.array(data, dtype=dt)
    if wavelet == 'beta':
        pass
    else:
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.r_[1:scales+1] * scalespace
    if data.ndim == 1:
        try:
            if wavelet.complex_cwt:
                out = np.zeros((np.size(scales), data.size), dtype=complex)
            else:
                out = np.zeros((np.size(scales), data.size))
        except AttributeError:
            out = np.zeros((np.size(scales), data.size))
        precision = 10
        if wavelet == 'beta':
            int_psi, x = BetaWavelet(betaParameters[0], betaParameters[1], betaParameters[2], betaParameters[3], betaParameters[4])
        else:    
            int_psi, x = integrate_wavelet(wavelet, precision=precision)
        step = x[1] - x[0]
        for i in np.arange(np.size(scales)):
            j = np.floor(
                np.arange(scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
            if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
            coef = - np.sqrt(scales[i]) * np.diff(np.convolve(data, int_psi[j.astype(int)][::-1]))
            d = (coef.size - data.size) / 2.
            out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
        #frequencies = scale2frequency(wavelet, scales, precision)
        #if np.isscalar(frequencies):
        #    frequencies = np.array([frequencies])
        #for i in np.arange(len(frequencies)):
        #    frequencies[i] /= sampling_period
        return out
    else:
        raise ValueError("Only dim == 1 supported")


def cwt_fixed_scipy(data, scales, wavelet, scalespace =1, sampling_period=1., betaParameters = [10000, beta_a,  beta_b, beta_cycles, 0]):
    
    '''
			Modified verstion of cwt_fixed() by Spencer Kirn
			COPIED AND FIXED FROM pywt.cwt TO BE ABLE TO USE WAVELET FAMILIES SUCH
			AS COIF AND DB
			
			COPIED From Spenser Kirn
			
			All wavelet work except bior family, rbio family, haar, and db1.
			
			cwt(data, scales, wavelet)
			
			One dimensional Continuous Wavelet Transform.
			
			Parameters
			----------
			data : array_like
				Input signal
			scales : array_like
				scales to use
			wavelet : Wavelet object or name
				Wavelet to use
			scalespace: integer
				If usingging spacing beteen scales, allowing quicker computation at lower resuolution	
			Returns
			-------
			coefs : array_like
			Continous wavelet transform of the input signal for the given scales
			and wavelet
			frequencies : array_like
			if the unit of sampling period are seconds and given, than frequencies
			are in hertz. Otherwise Sampling period of 1 is assumed.
			
			Notes
			-----
			Size of coefficients arrays depends on the length of the input array and
			the length of given scales.
			
			Examples
			--------
			>>> import pywt
			>>> import numpy as np
			>>> import matplotlib.pyplot as plt
			>>> x = np.arange(512)
			>>> y = np.sin(2*np.pi*x/32)
			>>> coef, freqs=pywt.cwt(y,np.arange(1,129),'gaus1')
			>>> plt.matshow(coef) # doctest: +SKIP
			>>> plt.show() # doctest: +SKIP
			----------
			>>> import pywt
			>>> import numpy as np
			>>> import matplotlib.pyplot as plt
			>>> t = np.linspace(-1, 1, 200, endpoint=False)
			>>> sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
			>>> widths = np.arange(1, 31)
			>>> cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')
			>>> plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
			...            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  # doctest: +SKIP
			>>> plt.show() # doctest: +SKIP
		'''
    dt = _check_dtype(data)
    data = np.array(data, dtype=dt)
    if wavelet == 'beta':
        pass
    else:
        if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
            wavelet = DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.r_[1:scales+1] * scalespace
    if data.ndim == 1:
        try:
            if wavelet.complex_cwt:
                out = np.zeros((np.size(scales), data.size), dtype=complex)
            else:
                out = np.zeros((np.size(scales), data.size))
        except AttributeError:
            out = np.zeros((np.size(scales), data.size))
        precision = 10
        if wavelet == 'beta':
            int_psi, x = BetaWavelet(betaParameters[0], betaParameters[1], betaParameters[2], betaParameters[3], betaParameters[4])
        else:    
            int_psi, x = integrate_wavelet(wavelet, precision=precision)
        step = x[1] - x[0]
        for i in np.arange(np.size(scales)):
            j = np.floor(
                np.arange(scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
            if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
            coef = - np.sqrt(scales[i]) * np.diff(ss.fftconvolve(data, int_psi[j.astype(int)][::-1]))
            d = (coef.size - data.size) / 2.
            out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
        #frequencies = scale2frequency(wavelet, scales, precision)
        #if np.isscalar(frequencies):
        #    frequencies = np.array([frequencies])
        #for i in np.arange(len(frequencies)):
        #    frequencies[i] /= sampling_period
        return out
    else:
        raise ValueError("Only dim == 1 supported")



def getThumbprint(data, wvt=WaveletToUse, ns=scales, scalespace = spacer, numslices=5, slicethickness=0.12, 
                  valleysorpeaks='both', normconstant=1, plot=False, betaParameters = [10000,2,5,2,3]):
    '''
    Updated version of the DWFT function above that allows plotting of just
    valleys or just peaks or both. To plot just valleys set valleysorpeaks='valleys'
    to plot just peaks set valleysorpeaks='peaks' or 'both' to plot both.
    '''
    # First take the wavelet transform and then normalize to one
    if np.shape(data)[0] == 2:
        wvt = data[1]
        data = data[0]
    
    #try:
    cfX = cwt_fixed(data, ns, wvt,scalespace,betaParameters=betaParameters)
    cfX = np.true_divide(cfX, abs(cfX).max()*normconstant)

    ns = np.shape(cfX)[0]

    fp = np.zeros((len(data), ns), dtype=int)

    # Create the list of locations between -1 and 1 to preform slices. Valley
    # slices will all be below 0 and peak slices will all be above 0.
    if valleysorpeaks == 'both':
        slicelocations1 = np.arange(-1 ,0.0/numslices, 1.0/numslices)
        slicelocations2 = np.arange(1.0/numslices, 1+1.0/numslices, 1.0/numslices)
        slicelocations = np.array(np.append(slicelocations1,slicelocations2))

    if valleysorpeaks == 'peaks':
        slicelocations = np.arange(1.0/numslices, 1+1.0/numslices, 1.0/numslices)

    if valleysorpeaks == 'valleys':
        slicelocations = np.arange(-1, 0.0/numslices, 1.0/numslices)

    for loc in slicelocations:
        for y in range(0, ns):
            for x in range(0, len(data)):
                if cfX[y, x]>=(loc-(slicethickness/2)) and cfX[y,x]<= (loc+(slicethickness/2)):
                    fp[x,y] = 1

    fp = np.transpose(fp[:,:ns])
    #except:
    #    fp = 'fail'
    
    return fp

def getThumbprint2(data, wvt=WaveletToUse, ns=scales, scalespace = spacer, numslices=5, slicethickness=0.12, 
                  valleysorpeaks='both', normconstant=1, plot=False):
    '''Modifications of DWFT code from Spencer Kirn and Margerat Rooney.  Calculates the thumbprint using
		   matrix math to the great extent possible allowing it to go faster than nested loops and comparisons.
		   The code does not allow for differences in positive and negative slice thickness, and has the same
		   thickeness for both white and dark bands.
		   
		   Inputs:
		   		data: the signal to be fingerprinted
		   		wvt: the wavelet that will be used
		   		ns = the number of scales that the 2d wavlet will be analyzed over
		   		scalespace = instead of doing every scall, you can skip and only calculate ever nth, with 
		   		      lower resolution, but more scales and quicker processing
		   		numslices = the number of white bands.  There will then be n-1 black bands.
		   	
		   	Output:
		   		2D matrix of 0's and 1's for the length of the data x the number of scales
    '''
    
        # First take the wavelet transform and then normalize to one
    if np.shape(data)[0] == 2:
        wvt = data[1]
        data = data[0]
    
    #Get the coefficents
    cfX = cwt_fixed_scipy(data, ns, wvt,scalespace)

    #normalize the coefficents to values between 0 and 1
    minVal = np.min(cfX)
    maxVal = np.max(cfX)
    
    highest = max([-minVal, maxVal])

    cfX /= highest


    #multiply by the number of slizes
    cfX *= float(numslices) * 1.5
    
    #truncate to integers
    cfX = np.matrix(cfX, dtype = int)

    #take modulous 2 so that every other integer is a value 1 or 0
    cfX = np.mod(cfX, 2).T
   
    return cfX

def RidgeCount(fingerprint):
    '''
    Count the number of times the fingerprint changes from 0 to 1 or 1 to 0 in 
    consective rows. Gives a vector representation of the DWFT
    '''
    diff = np.zeros((fingerprint.shape))
    
    for i, row in enumerate(fingerprint):
        if i==0:
            prev = row
        else:
            # First row (i=0) of diff will always be 0s because it does not
            # matter what values are present. 
            # First find where the rows differ
            diff_vec = abs(row-prev)
            # Then set those differences to 1 to be added later
            diff_vec[diff_vec != 0] = 1
            diff[i, :] = diff_vec
            
            prev = row
            
    ridgeCount = diff.sum(axis=0)
    
    return ridgeCount

def PlotFingerPrint(data, title = '', SaveSpot = location, ToSave = Saving, Show = True, Pdpi =PlotDPI, Titles=False):
    '''
    Makes an image of a fingerprint based on the Input data.  Does not calculate fingerprint with this funciton
    This functions takes the output of one of the makeFingerprint functions
    
    Inputs:
        data is a matrix of 0,1 from a fingerprint
        title is the title for the image, blank by default
        SaveSpot is where the image should be saved, if it is to be saved, default is the class initialization default loaction
        ToSave is whether the image should be just displated, or also saved, by default uses the Saving parameter
        
    Outputs:
        displayed or saved image of a fingerprint
    '''
    #FpScat=fp.getLabeledThumbprint(data, FP,scales,slices)
    #print(np.shape(data)[1], scales)

    
    #data = Input[0]
    #title = Input[1]
    
    scales = np.shape(data)[0]
    trim=0
    slices = 3
    #Show = True
    
    xName = np.arange(0,np.shape(data)[1]-2*trim,1)
    
    if trim == 0:
        Al,Ms  = np.meshgrid(xName,np.linspace(1,scales,scales))
    else:
        Al,Ms  = np.meshgrid(xName,np.linspace(1,scales,scales))

    

    fig1 = plt.figure(figsize=(PlotWidthIn,PlotHeightIn),dpi=Pdpi)
    ax1 = plt.axes()
    if trim == 0:
        cs1 = ax1.contourf(Al,Ms, data[:,:],cmap=my_cmap,levels=slices)
    else:
        cs1 = ax1.contourf(Al,Ms, data[:,trim:-trim],cmap=my_cmap,levels=slices)

    if Titles: plt.title(title)
    if ToSave: plt.savefig(SaveSpot+title.replace(" ", "").replace(":", "").replace(",", "").replace(".txt","").replace(".csv","")+FFormat)

    if Show: plt.show()
    else: plt.close(fig1)
        
    return 1

def getAcceleration(FileName):
    
        try:
    
            try:
                DataSet = np.genfromtxt(open(folder+FileName,'r'), delimiter=',',skip_header=0)
            except:
                DataSet = np.zeros((8,60000))
            JustFileName = FileName.rsplit('/', 1)[-1]
            if FileName[-20:-16] == 'Gyro':
                return [False,FileName,False]
            else:
                if FileName[-6:-5] == 's':
                    FileDate = FileName[-18:-7]
                    sensor = FileName[-5:-4]
                elif FileName[-21:-16] == 'Accel':
                    FileDate = FileName[-15:-4]
                    sensor = 1
                else:
                    FileDate = FileName[-20:-4]
                    sensor = 1
                return [[FileDate, 'x',DataSet[:,2], sensor,JustFileName],[FileDate,'y',DataSet[:,3],sensor,JustFileName],[FileDate,'z',DataSet[:,4],sensor,JustFileName]]
        except:
            return [False,FileName,False]

def butterHigh(data, cutFreq = 1000, frequency = 200000, order=5):
    '''
    Function to filter high frequency noise from a data signal, one of the options for the smoothing function
    
    Inputs:
    data : raw data to filter
    cutFreq : Frequency above which to filter
    frequency : frequency of the input signal / sample rate
    order : polynomial order for the butter function
    
    Output:
    Cleaned signal
    
    '''
    nyq = 0.5 * frequency
    normal_cutoff = cutFreq / nyq
    b, a = ss.butter(order, normal_cutoff, btype='highpass', analog=False)
    Clean = ss.filtfilt(b, a, data)
    return Clean

def butterBand(data, cutFreq = 1000, frequency = 200000, order=5):
    '''
    Function to filter high frequency noise from a data signal, one of the options for the smoothing function
    
    Inputs:
    data : raw data to filter
    cutFreq : Frequency above which to filter
    frequency : frequency of the input signal / sample rate
    order : polynomial order for the butter function
    
    Output:
    Cleaned signal
    
    '''
    nyq = 0.5 * frequency
    normal_cutoff = cutFreq / nyq
    b, a = ss.butter(order, normal_cutoff, btype='bandpass', analog=False)
    Clean = ss.filtfilt(b, a, data)
    return Clean

def low_pass_filter(data_in, wvt='sym2', dets_to_remove=5, levels=None):
    '''
    Function to filter out high frequency noise from a data signal. Usually 
    perform this before running the DWFT on the signal.
    
    data_in: input signal
    
    wvt: mother wavelet

    levels: number of levels to take in transformation

    dets_to_remove: details to remove in filter
    '''
    # vector needs to have an even length, so just zero pad if length is odd.
    if len(data_in) % 2 != 0:
        data_in = np.append(data_in, 0)
    
    coeffs = pywt.swt(data_in, wvt, level=levels)
    
    print(np.shape(coeffs))
    
    if levels is None:
        levels = len(coeffs)
    
    print(levels)
    
    for i in range(dets_to_remove):
        dets = np.asarray(coeffs[(levels-1)-i][1])
        dets[:] = 0
    
    filtered_signal = pywt.iswt(coeffs,wvt)
    return filtered_signal


def KalmanFilterDenoise(data, Kalrate=1):

    #https://jamwheeler.com/college-productivity/how-to-denoise-a-1-d-signal-with-a-kalman-filter-with-python/
    #

    def oavar(data, Kalrate, numpoints=30):

        x = np.cumsum(data)

        max_ratio = 1/9
        num_points = 30
        ms = np.unique(
            np.logspace(0, np.log10(len(x) * max_ratio), numpoints
           ).astype(int))        

        oavars = np.empty(len(ms))
        for i, m in enumerate(ms):
            oavars[i] = (
                (x[2*m:] - 2*x[m:-m] + x[:-2*m])**2
            ).mean() / (2*m**2)

        return ms / Kalrate, oavars

    def ln_NKfit(ln_tau, ln_N, ln_K):
        tau = np.exp(ln_tau)
        N, K = np.exp([ln_N, ln_K])
        oadev = N**2 / tau + K**2 * (tau/3)
        return np.log(oadev)

    def get_NK(data, Kalrate):
        taus, oavars = oavar(data, Kalrate)

        ln_params, ln_varmatrix = (
            curve_fit(ln_NKfit, np.log(taus), np.log(oavars))
        )
        return np.exp(ln_params)    

    # Initialize state and uncertainty
    state = data[0]
    output = np.empty(len(data))

    #rate = 1 # We can set this to 1, if we're calculating N, K internally
    # N and K will just be scaled relative to the sampling rate internally
    dt = 1/Kalrate
    
#    try:
    N, K = get_NK(data, Kalrate)
    process_noise = K**2 * dt
    measurement_noise = N**2 / dt

    covariance = measurement_noise


    for index, measurement in enumerate(data):
        # 1. Predict state using system's model

        covariance += process_noise

        # Update
        kalman_gain = covariance / (covariance + measurement_noise)

        state += kalman_gain * (measurement - state)
        covariance = (1 - kalman_gain) * covariance

        output[index] = state
    #except:
    #    output = np.ones(len(data))
    return output

def Smoothing(RawData, SmoothType = 1, SmoothDistance=15, wvt='sym2', Kalrate=1,
              dets_to_remove=5, levels=None, cutFreq = 1000, frequency = 200000, order=5):
    #Smooth type 0 or other is none
    #       type 1 is rolling average with SmoothDistance
    #       type 2 is low-filter denoise
    #       type 3 is Kalman filter

    if SmoothType ==1:
        SmoothedData = RawData
        if np.shape(np.shape(RawData))== 2:
            for i in range(SmoothDistance-1):
                for j in range(3):
                    SmoothedData[j,i+1]=np.average(RawData[j,0:i+1])
            for i in range(np.shape(RawData)[0]-SmoothDistance):
                for j in range(3):
                    SmoothedData[j,i+SmoothDistance]=np.average(RawData[j,i:i+SmoothDistance])
        elif np.shape(np.shape(RawData))== 1:
            for i in range(SmoothDistance-1):
                SmoothedData[i+1]=np.average(RawData[0:i+1])
            for i in range(np.shape(RawData)[0]-SmoothDistance):
                SmoothedData[i+SmoothDistance]=np.average(RawData[i:i+SmoothDistance])
    elif SmoothType == 2:
        SmoothedData = low_pass_filter(RawData, wvt, dets_to_remove, levels)
    elif SmoothType == 3:
        SmoothedData = KalmanFilterDenoise(RawData)
    elif SmoothType ==4:
        SmoothedData = butterHigh(RawData, cutFreq, frequency, order)
    elif SmoothType ==5:
        SmoothedData = butterBand(RawData, cutFreq, frequency, order)
    else:
        SmoothedData = RawData
    
    return SmoothedData

def getRAcceleration(Data):
    rVals = []
    for i in range(np.shape(Data)[0]):
        rVals.append(np.sqrt(Data[i,0]**2+Data[i,1]**2+Data[i,2]**2))
    return rVals

def truthVector(Filename):
    # Parses the filename, and compares it against the record of sensor position on cranes
    # inputs: filename
    # outputs: truth vector


    #Parsing the file name.  Assuming it is in the standard format
    sSensor = Filename[23]
    sDate = datetime.datetime.strptime('20'+Filename[10:21],"%Y%m%d-%H%M")

    mask = []

    i=0
    #loops through the known sensor movements, and creates a filter mask
    for spf in OutputVectors:
        
        startDate = datetime.datetime.strptime(str(spf[0])+str(spf[1]).zfill(2)+str(spf[2]).zfill(2)
            +str(spf[3]).zfill(2)+str(spf[4]).zfill(2),"%Y%m%d%H%M")
        #datetime.date(int(spf[0]), int(spf[1]), int(spf[2])) + datetime.timedelta(hours=spf[3]) + datetime.timedelta(minutes=spf[4])
        endDate = datetime.datetime.strptime(str(spf[5])+str(spf[6]).zfill(2)+str(spf[7]).zfill(2)
            +str(spf[8]).zfill(2)+str(spf[9]).zfill(2),"%Y%m%d%H%M")
        #datetime.date(int(spf[5]), int(spf[6]), int(spf[7])) + datetime.timedelta(hours=spf[8]) + datetime.timedelta(minutes=spf[9])
        
        if sDate >= startDate and sDate <= endDate and int(spf[10]) == int(sSensor):
            mask.append(True)
            i+=1
        else:
            mask.append(False)
        
    if i != 1: print('error ', i, Filename)

    results = OutputVectors[mask,11:]

    
    if i > 1: 
        print('Found Two ', Filename)
        results = results[0,:]
    #np.array(results)

    return results


def truthClass(Filename):
    # Parses the filename, and compares it against the record of sensor position on cranes
    # inputs: filename
    # outputs: truth vector


    #Parsing the file name.  Assuming it is in the standard format
    sSensor = Filename[23]
    sDate = datetime.datetime.strptime('20'+Filename[10:21],"%Y%m%d-%H%M")

    mask = []

    i=0
    #loops through the known sensor movements, and creates a filter mask
    for spf in OutputVectors:
        
        startDate = datetime.datetime.strptime(str(spf[0])+str(spf[1]).zfill(2)+str(spf[2]).zfill(2)
            +str(spf[3]).zfill(2)+str(spf[4]).zfill(2),"%Y%m%d%H%M")
        #datetime.date(int(spf[0]), int(spf[1]), int(spf[2])) + datetime.timedelta(hours=spf[3]) + datetime.timedelta(minutes=spf[4])
        endDate = datetime.datetime.strptime(str(spf[5])+str(spf[6]).zfill(2)+str(spf[7]).zfill(2)
            +str(spf[8]).zfill(2)+str(spf[9]).zfill(2),"%Y%m%d%H%M")
        #datetime.date(int(spf[5]), int(spf[6]), int(spf[7])) + datetime.timedelta(hours=spf[8]) + datetime.timedelta(minutes=spf[9])
        
        if sDate >= startDate and sDate <= endDate and int(spf[10]) == int(sSensor):
            mask.append(True)
            i+=1
        else:
            mask.append(False)
        
    if i != 1: print('error ', i, Filename)

    results = OutputVectors[mask,11:]
    results = np.array(results, dtype='bool')
    titles = OutputTitles[11:]
    result = titles[results]
    
    if i > 1: 
        print('Found Two ', Filename)
        results = results[0,:]
    #np.array(results)

    return result

def makeFrames(input, FrameLength = 600, numberFrames = 100): #,sequ,frameLength):
    frames=[] #np.array([],dtype=object,)
    segmentGap = int((np.shape(input)[1]-FrameLength)/numberFrames)
    #print(segmentGap,sequ, frameLength)
    for i in range(numberFrames):
        start = i * segmentGap
        imageMatrix = input[:,start:start+FrameLength]
        np.matrix(imageMatrix)
        imageMatrix = imageMatrix.T

        w0=int(np.shape(imageMatrix)[0]/2)
        a0=np.linspace(0,w0-1,w0,dtype=int)*2
        imageMatrix = np.delete(imageMatrix,a0,axis=0)

        w1=int(np.shape(imageMatrix)[1]/2)
        a1=np.linspace(0,w1-1,w1,dtype=int)*2
        imageMatrix = np.delete(imageMatrix,a1,axis=1)

        frames.append(imageMatrix)
    
    return frames

def ParseData(FPwMD):

        frames = np.asarray(makeFrames(FPwMD))

        #Results = truthVector(FPwMD[0])

        return frames #, Results
    
def RemoveNonmovers(Moves, files, XorAll = 'All'):

    maxes = np.amax(Moves[:,500:], axis = 1)
    mins = np.amin(Moves[:,500:], axis = 1)

    Keep = np.zeros(mins.size)
    fKeep = np.zeros(np.size(files))

    for i in range(mins.size):
        if i % 3 == 0:
            if maxes[i] > 0.01 and mins[i] < -0.01:
                Keep[i]=1
                if XorAll == 'All': Keep[i+1]=1
                if XorAll == 'All': Keep[i+2]=1
                fIndex = int(i/3)
                fKeep[fIndex] = 1
                

    Keep = np.array(Keep, dtype='bool')
    fKeep = np.array(fKeep, dtype='bool')
                
    SmallMoves = Moves[Keep,:]
    SmallFiles = list(compress(files, fKeep))

    return SmallMoves, SmallFiles

def SegmentMove(movement, CheckRange = 750):
    
    DataLength = np.size(movement)
    Segments = np.zeros(DataLength)

    for i in range(DataLength):
        if i < 100:
            cval = 0
        elif i<CheckRange:
            cval = sum(movement[0:i])
        else:
            cval = sum(movement[i-CheckRange:i+CheckRange])
        if cval > 50:
                Segments[i]=2
    return Segments

def PlotColorScales(Input, title = 'None Given', Show = True, trim = 0, saveSpot = location):
    
    #FpScat=fp.getLabeledThumbprint(data, FP,scales,slices)
    #print(np.shape(data)[1], scales)

    
    data = Input[0]
    title = Input[1]
    
    scales = np.shape(data)[0]
    #trim=0
    #slices = 3
    #Show = True
    
    xName = np.arange(0,np.shape(data)[1]-2*trim,1)
    
    if trim == 0:
        Al,Ms  = np.meshgrid(xName,np.linspace(1,scales,scales))
    else:
        Al,Ms  = np.meshgrid(xName,np.linspace(1,scales,scales))

    

    fig1 = plt.figure(figsize=(PlotWidthIn,PlotHeightIn),dpi=PlotDPI)
    ax1 = plt.axes()
    if trim == 0:
        cs1 = ax1.contourf(Al,Ms, data[:,:],cmap='jet',levels=256)
    else:
        cs1 = ax1.contourf(Al,Ms, data[:,trim:-trim],cmap='jet',levels=256)

    if Titles: plt.title(title)
    if Saving: plt.savefig(saveSpot+title.replace(" ", "").replace(":", "").replace(",", "").replace(".txt","")+FFormat)

    if Show: plt.show()
    else: plt.close()
        
    return 1

def getScalesOnly(data, wvt=WaveletToUse, ns=scales, scalespace = spacer, plot=False):
    '''Attempt to speed code where the comparisons happen too many times too slowly
    '''
    if np.shape(data)[0] == 2:
        wvt = data[1]
        data = data[0]
    
    #try:
    cfX = cwt_fixed_scipy(data, ns, wvt,scalespace)
    mins = np.min(cfX)
    maxs = np.max(cfX)
    cfX -= mins
    cfX *= 1/(maxs-mins)
    cfX *= 255

    return cfX

def KalmanGroup(DataMatrix):
    waveKalmaned = np.asarray([],dtype=object)
    waveKalmaned = Parallel(n_jobs=num_cores)(delayed(KalmanFilterDenoise)(np.asarray(data).flatten()) for data in DataMatrix)
    waveKalmaned = np.matrix(waveKalmaned)
    length = np.shape(waveKalmaned)[0]
    justifier = np.ones((length, np.shape(waveKalmaned)[1]))
    average = np.zeros(length)
    for i in range(length):
        average[i]= np.average(waveKalmaned[i][:])
    justifier = justifier.T * average.T
    waveKalmaned = waveKalmaned - justifier.T
    
    return waveKalmaned

def makeMatrixImages(DataMatrix, wvt = WaveletToUse, scales = 1000, spacer = 1):

    xPrint = getScalesOnly(np.asarray(DataMatrix[0]).flatten(), wvt, scales, spacer)
    yPrint = getScalesOnly(np.asarray(DataMatrix[1]).flatten(), wvt, scales, spacer)
    zPrint = getScalesOnly(np.asarray(DataMatrix[2]).flatten(), wvt, scales, spacer)

    PrintMatrix = np.dstack((xPrint,yPrint,zPrint))
    
    return np.asarray(PrintMatrix)

def makeMatrixPrints(DataMatrix, wvt = WaveletToUse):

    xPrint = getThumbprint(np.asarray(DataMatrix[0]).flatten(), wvt)*255
    yPrint = getThumbprint(np.asarray(DataMatrix[1]).flatten(), wvt)*255
    zPrint = getThumbprint(np.asarray(DataMatrix[2]).flatten(), wvt)*255

    PrintMatrix = np.dstack((xPrint.T,yPrint.T,zPrint.T))
    
    return np.asarray(PrintMatrix)

def makeMPFast(DataMatrix, wvt = WaveletToUse, scales = 1000, spacer = 1, title = ''):
    '''
        Makes a 3D thumbprint images from a 3D motion source.
        The XYZ components of Acceleration each are turned into a thumbprint
        and the thumbprints are then stored as the RBG colors for a single image.

        Inputs:
            DataMatrix: 3 arrays of XYZ components of a signal
            wvt: The wavelet used to make the DWFP
            sacales: number of scales to using in makig the wavelet
            spacer: weather or not the scales are consecutive integers, or skip numbers
            title: if a title is given, the image will be saved as a 
                    png with that file name, if not given, not saved
        Outputs:
            array of 3 arrays that can be an 0-255 color depth image

        Notes:
            other getThumprint2 inputs are left at defaults, such as scales,
            scale spacing, etc.
    
    '''
    xPrint = getThumbprint2(np.asarray(DataMatrix[0]).flatten(), wvt, scales, spacer)*255
    yPrint = getThumbprint2(np.asarray(DataMatrix[1]).flatten(), wvt, scales, spacer)*255
    zPrint = getThumbprint2(np.asarray(DataMatrix[2]).flatten(), wvt, scales, spacer)*255

    PrintMatrix = np.dstack((np.asarray(xPrint.T),np.asarray(yPrint.T),np.asarray(zPrint.T)))

    if len(title)> 1:
        cv2.imwrite(title + '.png', PrintMatrix)
    
    return np.asarray(PrintMatrix)
