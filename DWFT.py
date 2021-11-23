#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt._extensions._pywt import (DiscreteContinuousWavelet, ContinuousWavelet,
                                Wavelet, _check_dtype)
from pywt._functions import integrate_wavelet, scale2frequency
from skimage.measure import label
from skimage.morphology import convex_hull_image
from time import time as ti

def cwt_fixed(data, scales, wavelet, sampling_period=1.):
    """
    COPIED AND FIXED FROM pywt.cwt TO BE ABLE TO USE WAVELET FAMILIES SUCH
    AS COIF AND DB
    
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

    # accept array_like input; make a copy to ensure a contiguous array
    dt = _check_dtype(data)
    data = np.array(data, dtype=dt)
    if not isinstance(wavelet, (ContinuousWavelet, Wavelet)):
        wavelet = DiscreteContinuousWavelet(wavelet)
    if np.isscalar(scales):
        scales = np.array([scales])
    if data.ndim == 1:
        try:
            if wavelet.complex_cwt:
                out = np.zeros((np.size(scales), data.size), dtype=complex)
            else:
                out = np.zeros((np.size(scales), data.size))
        except AttributeError:
            out = np.zeros((np.size(scales), data.size))
        for i in np.arange(np.size(scales)):
            precision = 10
            int_psi, x = integrate_wavelet(wavelet, precision=precision)
            step = x[1] - x[0]
            j = np.floor(
                np.arange(scales[i] * (x[-1] - x[0]) + 1) / (scales[i] * step))
            if np.max(j) >= np.size(int_psi):
                j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
            coef = - np.sqrt(scales[i]) * np.diff(
                np.convolve(data, int_psi[j.astype(np.int)][::-1]))
            d = (coef.size - data.size) / 2.
            out[i, :] = coef[int(np.floor(d)):int(-np.ceil(d))]
        frequencies = scale2frequency(wavelet, scales, precision)
        if np.isscalar(frequencies):
            frequencies = np.array([frequencies])
        for i in np.arange(len(frequencies)):
            frequencies[i] /= sampling_period
        return out, frequencies
    else:
        raise ValueError("Only dim == 1 supported")


def low_pass_filter(data_in, wvt, dets_to_remove, levels=None):
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
    
    if levels is None:
        levels = len(coeffs)
    
    for i in range(dets_to_remove):
        dets = np.asarray(coeffs[(levels-1)-i][1])
        dets[:] = 0
    
    filtered_signal = pywt.iswt(coeffs,wvt)
    return filtered_signal

def DWFT(data_in, wvt, ns=50, numridges=5, rthickness=0.12, plot=True):
    '''
     Create the wavelet fingerprint
     data_in: raw 1D signal in which the wavelet thumbprint is created
     wvt: name of mother wavelet options shown in pywt.wavelist(kind='continuous)
     ns: number of scales to use in the continuous wavelet transform (start with 50)
         must be less then the length of data_in
     numridges: number of ridges used in the wavelet thumbprint (start with 5)
     rthickness: thickness of the ridges normalized to 1 (start with 0.12)
     plot: Boolean

    '''
    
    cfX, freqs = cwt_fixed(data_in, np.arange(1,ns+1), wvt)
    # cfX is a ns by len(data_in) matrix with each value corresponding to a 
    # wavelet coefficient for that specific time and scale
    # Now normalize based on the largest absolute coefficient
    cfX = np.true_divide(cfX, np.amax(np.amax(np.absolute(cfX)))) 
    
    # Create square matrix of zeros to initialize thumbprint
    thumbprint = np.zeros((len(data_in), ns), dtype=int)
    
    rlocations1 = np.arange(-1 ,0.0/numridges, 1.0/numridges)
    rlocations2 = np.arange(1.0/numridges, 1+1.0/numridges, 1.0/numridges)
    rlocations = np.array(np.append(rlocations1,rlocations2))
    
    for sl in range(0, len(rlocations)):
        for y in range(0, ns):
            for x in range(0, len(data_in)):
                if cfX[y,x]>=(rlocations[sl]-(rthickness/2)) and cfX[y,x]<=(rlocations[sl]+(rthickness/2)):
                    thumbprint[x,y] = 1
                    
    thumbprint = np.transpose(thumbprint[:,:ns])
    
    if plot:
        plt.figure()
        plt.imshow(thumbprint, cmap='gray', interpolation='nearest', aspect='auto')
        #plt.colorbar(ticks=(0,0.5,1))
        plt.ylim((ns-1.0))
        plt.xlim((0,len(data_in)))
        #plt.tight_layout()
        plt.show()
    
    return thumbprint

def getThumbprint(data, wvt, ns=50, numslices=5, slicethickness=0.12, 
                  valleysorpeaks='both', normconstant=1, plot=True):
    '''
    Updated version of the DWFT function above that allows plotting of just
    valleys or just peaks or both. To plot just valleys set valleysorpeaks='valleys'
    to plot just peaks set valleysorpeaks='peaks' or 'both' to plot both.
    '''
    # First take the wavelet transform and then normalize to one
    cfX, freqs = cwt_fixed(data, np.arange(1,ns+1), wvt)
    cfX = np.true_divide(cfX, abs(cfX).max()*normconstant)
    
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
    
    if plot:
        plt.figure()
        plt.imshow(fp, cmap='gray', interpolation='nearest', aspect='auto')
        plt.ylim((ns-1.0))
        plt.xlim((0, len(data)))
        plt.show()
        
    return fp

def FPFeatureVector(fp, obj=None, normHOG=True, HOGunsigned=True):
    '''
    normHOG: bool. if True normalize HOG values by the time duration of the
            fingerprint
    The returned feature vector has features in this order:
        0. semimajor axis
        1. semiminor axis
        2. theta (angle of orientation for ellipse)
        3. eccentricity
        4. degree 2 polynomial a
        5. degree 2 polynomial b (ax^2 + bx +c)
        6. degree 2 polynomial c
        7. degree 4 polynomial a
        8. degree 4 polynomial b
        9. degree 4 polynomial c
        10. degree 4 polynomial d (ax^4 + bx^3 + cx^2 + dx + e)
        11. degree 4 polynomial e
        12. HOGAngle0
        13. HOGAngle45
        14. HOGAngle90
        15. HOGAngle135
        16. wavelet scale value for FP center of mass
        17. number of time steps for fingerprint
        18. diameter of circle with same area as fingerprint
        19. Extent
        20. Area
        21. Filled Area
        22. Euler Number
        23. Convex Area
        24. Solidity
    '''
    if obj is None:
        P = getObjPMat(fp, _largestFP(fp))
    
    else:
        P = getObjPMat(fp, obj)
    
    # Create a temporary fingerprint of size fp that only has values for P
    tempFP = np.zeros(fp.shape)
    for point in P:
        tempFP[int(point[1]), int(point[0])] = 1
    
    com = _objCenterMass(P)
    
    # First get the properties of an ellipse that would most closely match the fp
    featVec = np.zeros((4,))
    featVec[0], featVec[1], featVec[2], featVec[3] = _ellipseProperties(P)
    
    # Fit polynomials of degree 2 and 4 to outside of fp
    deg2Poly, deg4Poly = _polyFit(P, fp.shape[0])
    
    # Subtract center of mass from y intercept in polynomial fits
    deg2Poly[-1] -= com[0]
    deg4Poly[-1] -= com[0]
    
    featVec = np.append(featVec, deg2Poly)
    featVec = np.append(featVec, deg4Poly)
    
    if HOGunsigned:
        num_bins = 4
    else:
        num_bins = 8 
    HogFeatures, angleDict, image = FpHogFeats(P, fp.shape[0], window=None, 
                                               num_bins=num_bins, 
                                               unsigned=HOGunsigned)
    
    fpTime = P[:, 0].max() - P[:, 0].min()
    if normHOG:
        HogFeatures /= fpTime
        
    featVec = np.append(featVec, HogFeatures)
    # Finally add in the center of mass wavelet scale, the length in time of fp
    # and the diameter of a circle of equivilent area.
    featVec = np.append(featVec, com[1])
    featVec = np.append(featVec, fpTime)
    featVec = np.append(featVec, _diameter(P))
    featVec = np.append(featVec, _extent(P))
    featVec = np.append(featVec, _objArea(P))
    featVec = np.append(featVec, _filledArea(tempFP, P))
    featVec = np.append(featVec, _eulerNumber(tempFP))
    featVec = np.append(featVec, _convexHullArea(tempFP))
    featVec = np.append(featVec, _solidity(tempFP))

    return featVec, image

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

def _largestFP(fp):
    '''
    This function will select the largest fingerprint in a wavelet thumbprint
    '''
    totalObjs = fp.max()+1
    
    mostPix = 0
    for i in range(1, totalObjs):
        totPix = len(fp[fp==i])
        if totPix > mostPix:
            mostPix = totPix
            bigObj = i
            
    return bigObj

def _isSurrounded(array, entityLabel):
    '''
    This function finds all entries in a fingerprint that correspond with a 
    specific label from the skimage.metrics label function. If all the nearest
    non-zero entries to a labeled section are the same, that section is surrounded
    by another section and they receive the same label.
    '''
    locs = np.argwhere(array == entityLabel)
    
    # Test three locations, directly above, directly to the right and directly
    # to the left. if all three have the same value then the entity is said
    # to be surrounded
    testPt = np.array(locs[np.random.randint(0, high=len(locs)), :])
    
    # Test Above
    aboveVal = array[testPt[0], testPt[1]]
    while aboveVal == 0 or aboveVal == entityLabel:
        # Concession if we hit the top of the image
        if testPt[0] == 0:
            break
        testPt[0] -= 1
        aboveVal = array[testPt[0], testPt[1]]
    
    # Test to the right
    testPt = np.array(locs[np.random.randint(0, high=len(locs)), :])
    count = 0
    while testPt[1] == array.shape[1]-1 and count < 5:
        testPt = np.array(locs[np.random.randint(0, high=len(locs)), :])
        count += 1
        
    rightVal = array[testPt[0], testPt[1]]
    while rightVal == 0 or rightVal == entityLabel:
        testPt[1] += 1
        if testPt[1] >= array.shape[1]-1:
            rightVal = 'edge'
            break
        rightVal = array[testPt[0], testPt[1]]
        # Concession if we hit end of array
    
    testPt = np.array(locs[np.random.randint(0, high=len(locs)), :])
    count = 0
    while testPt[1] == 0 and count < 5:
        testPt = np.array(locs[np.random.randint(0, high=len(locs)), :])
        count += 1
    
    leftVal = array[testPt[0], testPt[1]]
    while leftVal == 0 or leftVal == entityLabel:
        # Concession if we hit end of array
        if testPt[1] == 0:
            leftVal = 'edge'
            break
        
        testPt[1] -= 1
        leftVal = array[testPt[0], testPt[1]]
        
    if aboveVal == leftVal and leftVal == rightVal:
        return aboveVal

    elif leftVal == 'edge':
        if aboveVal == rightVal:
            return aboveVal
        
    elif rightVal == 'edge':
        if aboveVal == leftVal:
            return aboveVal
        
    else:
        return None

def _fixLabels(fp):
    '''
    This function adjusts the labels of the labeled fingerprint so there are no
    integer values skipped. 
    '''
    unique = np.unique(fp)[1:]
    new_labels = list(range(1, len(unique)+1))
    
    for old_lab, new_lab in zip(unique, new_labels):
        fp[fp==old_lab] = new_lab

def getLabeledThumbprint(data, wavelet, ns=50, numslices=5, slicethickness=0.12,
                         valleysorpeaks='both'):
    '''
    This function creates two fingerprints, one for the peaks of the wavelet
    coefficient matrix and another for the valleys. This allows more separation 
    between the fingerprints on the plot. More separation allows for more accurate
    labeling because we will not have to worry about two fingerprints overlapping.
    Once these fingerprints are made a correction function ensures that all
    fingerprints are correctly labeled by checking if certain parts of a fingerprint
    are surrounded by another. If one labeled entity is surrounded by another
    then they are considered to be the same fingerprint and labeled accordingly.
    
    '''
    
    if valleysorpeaks == 'both' or valleysorpeaks =='peaks':
        labeledfpPeaks = getLabeledFP(data, wavelet, ns, numslices, slicethickness,
                                      'peaks')
        
    if valleysorpeaks == 'both' or valleysorpeaks == 'valleys':
        labeledfpValleys = getLabeledFP(data, wavelet, ns, numslices, slicethickness,
                                        'valleys')
    
    # shift labels in peaks matrix to ensure unique labels
    if valleysorpeaks == 'both':
        maxLabel = labeledfpPeaks.max()
        labeledfpValleys[labeledfpValleys!=0] += maxLabel
    
        # add together to create one single fingerprint
        labeledFP = labeledfpValleys + labeledfpPeaks
       
    elif valleysorpeaks == 'peaks':
        labeledFP = labeledfpPeaks
        
    elif valleysorpeaks == 'valleys':
        labeledFP = labeledfpValleys
        
    return labeledFP

def getLabeledFP(data, wavelet, ns=50, numslices=5, slicethickness=0.12, 
                 valleysorpeaks='peaks'):
    '''
    Function to label just the peaks or just the valleys of a wavelet 
    fingerprint. If you want to label both peaks and valleys use 
    getLabeledThumbprint function
    '''
    fp = getThumbprint(data, wavelet, ns, numslices, slicethickness, 
                             valleysorpeaks=valleysorpeaks, plot=False)
    labeledfp = label(fp, background=0, connectivity=2)
    
    # Make sure valleys and peaks are correctly labeled
    for i in range(1, labeledfp.max()+1):
        result = 0
        labelOfInterest = i
        
        if len(np.argwhere(labeledfp == labelOfInterest)) > 0:        
            while result != None:
                result = _isSurrounded(labeledfp, labelOfInterest)
                if result != None:
                    labeledfp[labeledfp == labelOfInterest] = result
                    labelOfInterest = result
    
    return labeledfp

def HOGFeatures(image, num_bins=4, window=8, unsigned=True):
    '''
    Creates the HOG feature vector for the input image.
    
    num_bins: default=4
        int. Number of bins to use between 0-180 (if unsigned=True) or -180-180 
        (if unsigned=False)
        
    window: default=8
        int or None, if int: Size of window for each histogram. Each histogram 
        is defined for a window x window submatrix of the image. if None: 
        then functions returns a vector of size num_bins, which is a histogram
        of the entire image.
        
    unsigned: default=True
        bool, Defines interval of angles. If True interval is 0-180, else
        -180-180
    '''
    G, theta = _getGradient(image, unsigned)
    
    if window != None:          
        lr_steps = int(np.floor(image.shape[1]/window))
        ud_steps = int(np.floor(image.shape[0]/window))
    else:
        lr_steps = None
        ud_steps = None
    
    if unsigned:
        angleStep = 180/num_bins
        angleDict = {i*angleStep:i for i in range(num_bins)}
    else:
        angleStep = 360/num_bins
        angleDict = {(i*angleStep)-180:i for i in range(num_bins)}
        
    hist = _getHoGHist(G, theta, lr_steps, ud_steps, angleDict, num_bins, window, unsigned)
    
    return hist, angleDict

def FpHogFeats(P, ns, num_bins=4, window=8, unsigned=True):
    
    hogP = np.zeros(P.shape)
    for i, line in enumerate(P):
        hogP[i, 0] = line[0] - P[:, 0].min()
        hogP[i, 1] = line[1]
        
    image = np.zeros((ns, int(hogP[:, 0].max()+1)))
    for point in hogP:
        image[int(point[1]), int(point[0])] = 1
    
    hist, angleDict = HOGFeatures(image, num_bins, window, unsigned)
    
    return hist, angleDict, image
    
def _dropSmallObjs(fp, threshold=50):
    '''
    This function iterates through all the different objects in a labeled fingerprint
    and omits any object that has less pixels then the defined threshold. The
    number of pixels only includes pixels that have a non-zero value (this will
    correspond to the objects label number).
    '''
    
    for i in range(1, fp.max()+1):
        if len(fp[fp==i]) < threshold:
            fp[fp==i] = 0
            
def getProcessedFP(vec, wvt, ns=50, numslices=5, slice_thickness=0.12, threshold=50):
    
    fp = getLabeledThumbprint(vec, wvt, ns=ns, numslices=numslices, 
                              slicethickness=slice_thickness)
    
    _dropSmallObjs(fp, threshold)
    
    _fixLabels(fp)
    
    plt.figure()
    plt.subplot(211)
    plt.imshow(fp, cmap='gray', interpolation='nearest', aspect='auto')
    
    ridgecount = RidgeCount(fp)
    
    plt.subplot(212)
    (markerline, stemlines, baseline) = plt.stem(ridgecount, linefmt='k:', markerfmt='ko')
    plt.setp(baseline, visible=False)
    plt.xlim([0, len(ridgecount)])
    plt.tight_layout()
    
    plt.subplot(211)
    
    #fp_CoM is a matrix with the center of mass of all fingerprints. The first
    # line is intentionally left as 0,0 so every other line corresponds with the
    # object label number
    fp_CoM = np.zeros((fp.max()+1, 2))
    for i in np.unique(fp):
        if i != 0:
            P = getObjPMat(fp, i)
            
            allVals = _allPixels(P, fp.shape[0])
            
            CoM = _objCenterMass(allVals)
            plt.scatter([int(round(CoM[0]))], [int(round(CoM[1]))], marker='x',
                         color='g', s=100)
            fp_CoM[i,:] = CoM
    
    return fp, fp_CoM

def getObjPMat(fp, obj):
    '''
    The P matrix is an Ax2 matrix that gives the scale value and the translation
    value of every point that has the label obj. To stay consistent with
    published materials (see: Bertoncini and Dieckman theses) the P[:,0] is set
    to be the b coordinate in the I(a,b) matrix and P[:,1] is the a coordinate,
    where I is the labeled wavelet fingerprint a is the wavelet scale value and 
    b is the translation value.
    '''
    temp = np.argwhere(fp==obj)
    
    #Flip the columns to stay consistent with Bertoncini and Dieckman
    P = np.zeros(temp.shape)
    P[:,0] = temp[:,1]
    P[:,1] = temp[:,0]
    
    return P

def _objArea(a):
    '''
    Finds the total number of pixels that have the label of obj. Input is the 
    result of either the getAllPixels or getObjPMat function
    '''
    return a.shape[0]

def _objCenterMass(a):
    '''
    Function to find the center of mass of a labeled object in a wavelet 
    fingerprint. Input is the P matrix resulting from either the getAllPixels 
    or getObjPMat function.
    '''
    return a.sum(axis=0)/a.shape[0]

def _unique_rows(a):
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx = np.unique(b, return_index=True)
    unique_a = a[idx]
    return unique_a

def _outsideVals(P, ns):
    '''
    This function locates all the outermost values of a specific fingerprint.
    '''
    scales = range(int(P[:,1].min()), int(P[:,1].max())+1)
    time = range(int(P[:,0].min()), int(P[:,0].max())+1)
    
    # P gives the coordinates (time, scale) of every non-zero point in the 
    # fingerprint.
    P_scales = P[:,1]
    P_time = P[:,0]
    
    OutsideVals = []
    for row in scales:
        # Locate all points on a row that correspond to the FP then record the 
        # outermost
        temp = np.argwhere(P_scales == row)
        if len(temp) == 0:
            continue
        
        OutsideVals.append(P[temp[:,0].min(), :])
        OutsideVals.append(P[temp[:,0].max(), :])
    
    if P_scales.max() == ns - 1:
        max_scale = False
    else:
        max_scale = True
    for col in time:
        temp = np.argwhere(P_time == col)
        if len(temp) == 0:
            continue
        OutsideVals.append(P[temp[:, 0].min(), :])
        if max_scale:
            OutsideVals.append(P[temp[:, 0].max(), :])
        
    return _unique_rows(np.array(OutsideVals))
    
def _startingPoint(outsideVals, time):
    '''
    This function finds the index of the point to start with when finding all
    pixel values inside a fingerprint.
    
    time --> integer that gives coordinate on the x axis of the fingerprint
    '''
    idx = np.argwhere(outsideVals[:,0] == time)
    # A lot of weird Python things with array shapes in this function. Get used to it
    idx = idx.reshape((idx.shape[0],))
    
    t_locs = outsideVals[idx,:]
    t_locs = t_locs.reshape((t_locs.shape[0], t_locs.shape[1],))
    
    idx = int(np.argwhere(t_locs[:,1] == t_locs[:,1].min()))
    startPoint = t_locs[idx, :]
    
    return startPoint

def _leftRight(outsideVals, point):
    '''
    Given a pixel on an fingerprint and all the outer points of that fingerprint
    this function will tell if that point falls within that fingerprint
    '''
    idx = list(np.argwhere(outsideVals[:, 1] == point[0][1]))
    sideVals = outsideVals[idx, :]
    toLeft = False
    toRight = False
    
    for row in sideVals:
        if row[0][0] < point[0][0]:
            toLeft = True
        if row[0][0] > point[0][0]:
            toRight = True
        
    if toLeft and toRight:
        return True
    else:
        return False

def _allPixels(P, ns):
    '''
    This function takes the outside values found using the getOutsideVals function
    and finds all points of the fingerprint within those outside values.
    '''
    outsideVals = _outsideVals(P, ns)
    allVals = np.array(outsideVals)
    uniqueTimeVals = np.unique(outsideVals[:, 0])
    
    allVals = np.append(allVals, outsideVals).reshape((-1,2))
    
    for i, t in enumerate(uniqueTimeVals):
        point = _startingPoint(outsideVals, t)
        point = point.reshape((1,2))
        while point[0][1] < ns:
            point[0][1] += 1
            inside = _leftRight(outsideVals, point)
            if inside:
                allVals = np.append(allVals, point, axis=0)
                
    return _unique_rows(allVals)

def MakeEllipse(semimajor, semiminor, theta, cb, ca, num_points=1000):
    '''
    This function returns the data to plot an ellipse overtop of a wavelet 
    fingerprint. To be used to check the ellipse to ensure it is correct.
    
    plt.plot(data[0], data[1])
    '''
    angles = np.linspace(0, 2*np.pi, num_points)
    r = 1/np.sqrt((np.cos(angles))**2 + (np.sin(angles))**2)
    x = r*np.cos(angles)
    y = r*np.sin(angles)
    data = np.array([x,y])
    S = np.array([[semimajor, 0], [0, semiminor]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    T = np.dot(R,S)
    data = np.dot(T, data)
    data[0] += cb
    data[1] += ca
    
    return data

def _moment(P, p, q):
    '''
    Returns the M_p,q moment of the fingerprint described by P. This only works
    because the pixels in each fingerprint are defined to be 1.
    '''
    return np.multiply((P[:,0]**p), (P[:,1]**q)).sum()

def _ellipseProperties(P):
    
    M00 = _moment(P, 0, 0)
    M10 = _moment(P, 1, 0)
    M01 = _moment(P, 0, 1)
    M11 = _moment(P, 1, 1)
    M20 = _moment(P, 2, 0)
    M02 = _moment(P, 0, 2)
    # calculate central point
    x_bar = M10/M00
    y_bar = M01/M00
    # get intermediate values
    mu20 = M20/M00 - x_bar**2
    mu11 = 2*(M11/M00 - x_bar*y_bar)
    mu02 = M02/M00 - y_bar**2

    tanTerm = mu11/(mu20-mu02)
    theta = 0.5*np.arctan(tanTerm)
    # angle correction if mu20<mu02 (same as inverting tanTerm)
    if mu20<mu02:
        theta += np.pi/2
    
    gamma = np.sqrt((mu11**2) + (mu20-mu02)**2)
    semimajor = np.sqrt(8 * (mu20 + mu02 + gamma))/2
    semiminor = np.sqrt(8 * (mu20 + mu02 - gamma))/2
    
    ecc = np.sqrt(1-((semiminor**2)/(semimajor**2)))
    
    return semimajor, semiminor, theta, ecc

def _diameter(P):
    '''
    This function calculates the diameter of a circle with the same area as
    the fingerprint
    '''
    A = _objArea(P)
    return np.sqrt((4*A)/np.pi)

def _polyFit(P, ns):
    
    outerVals = _outsideVals(P, ns)
    
    # Polyfit will not work if there are more then one scale point at a specific
    # time value so find the lowest scale value for each
    timeRange = range(int(outerVals[:,0].min()), int(outerVals[:,0].max()))
    for i, t in enumerate(timeRange):
        tVals = outerVals[outerVals[:,0]==t]
        if len(tVals) == 0:
            continue
        
        if tVals.shape[1]>1:
            val = tVals[tVals[:,1]==tVals[:,1].min()]
        else:
            val = tVals
        if i==0:
            polyVals = val
        else:
            polyVals = np.append(polyVals, val, axis=0)
    
    deg2_fit = np.polyfit(polyVals[:,0]-polyVals[:, 0].mean(), polyVals[:,1], deg=2)
    deg4_fit = np.polyfit(polyVals[:,0]-polyVals[:, 0].mean(), polyVals[:,1], deg=4)
    
    return (deg2_fit, deg4_fit)

def _filledArea(fp, P):
    '''
    Calculates the total area of a fingerprint, if all the holes were filled
    '''
    outerVals = _outsideVals(P, fp.shape[1])
   
    filled = np.array(fp)
    tSteps = np.unique(outerVals[:,0])
    for val in tSteps:
        val = int(val)
        pts_inCol = outerVals[outerVals[:,0] == val]
        top = pts_inCol[:, 1].min()
        bot = pts_inCol[:, 1].max()
        filled[int(top):int(bot), val] = 1
    
    return filled.sum()

def _boundingBoxArea(P):
    '''
    Calculates the area of the smallest box that can contain the entire
    fingerprint. Adds one to both height and width because bounding box must
    contain entire fingerprint, not bisect it.
    '''
    width = (max(P[:,1]) - min(P[:, 1])) + 1
    height = (max(P[:, 0]) - min(P[:, 0])) + 1
    
    return height * width

def _extent(P):
    '''
    Extent gives the ratio of the total pixels in the fingerprint to the total
    number of pixels in a bounding box of the fingerprint    
    '''
    return _objArea(P)/_boundingBoxArea(P)

def _eulerNumber(fp):
    '''
    Calculates the number of Q1, Q3 and QD bit quads in a wavelet fingerprint
    to calculate the Euler Number:
        E = 1/4 (n{Q1} - n{Q3} - n{QD})
    where a Q1 bit quad is a four pixel area with one "on" pixel, a Q3 bit quad 
    is a four pixel area with three "on" pixels, and QD is a four pixel area
    with "on" pixels on either diagonal. (Pratt, Digital Image Processing)
    '''
    oneQuads = 0
    threeQuads = 0
    diagQuads = 0
    
    # zero pad the fingerprint array (This is what they did in MatLab bweuler fcn)
    array = np.array(fp)
    array = np.append(np.zeros((array.shape[0], 1)), array, axis=1)
    array = np.append(array, np.zeros((array.shape[0], 1)), axis=1)
    array = np.append(array, np.zeros((1, array.shape[1])), axis=0)
    array = np.append(np.zeros((1, array.shape[1])), array, axis=0)
    
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            
            bitQuad = array[i:i+2, j:j+2]
            # binary image, so sum gives total "on" pixels
            bitSum = bitQuad.sum()
            
            if bitSum == 1:
                oneQuads += 1
                
            elif bitSum == 3:
                threeQuads += 1
            
            elif bitSum == 2:
                if bitQuad[0,0] == 1 and bitQuad[1,1] == 1:
                    diagQuads += 1
                    
                elif bitQuad[1,0] == 1 and bitQuad[0,1] == 1:
                    diagQuads += 1
                    
    return (1/4)*(oneQuads - threeQuads - 2*diagQuads)
    
def _convexHullArea(fp):
    '''
    This function uses skimage to calculate the convex hull image area for the 
    given fingerprint fp. First is must be made C_CONTIGUOUS if it is not already
    '''
    array = np.ascontiguousarray(fp)
    
    hullArray = convex_hull_image(array)
    
    return hullArray.sum()

def _solidity(fp):
    return fp.sum()/_convexHullArea(fp)   
    
def _getGradient(image, unsigned):
    filt = np.array([-1,0,1])
    
    G = np.zeros(image.shape)
    theta = np.zeros(image.shape)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if j != 0 and j != image.shape[1] - 1:
                g_x = np.dot(filt, image[i, j-1:j+2])
                
            elif j == 0:
                g_x = np.dot(filt, np.append(image[i, 0], image[i, j:j+2]))
                
            elif j == image.shape[0] - 1:
                g_x = np.dot(filt, np.append(image[i, j-1:j+1], image[i, -1]))
                
            if i != 0 and i != image.shape[0] - 1:
                g_y = np.dot(filt, image[i-1:i+2, j])
                
            elif i == 0:
                g_y = np.dot(filt, np.append(image[0, j], image[i:i+2, j]))
                
            elif i == image.shape[1]:
                g_y = np.dot(filt, np.append(image[i-1:i+1, j], image[-1, j]))
             
            if unsigned:
                G[i, j] = abs(np.sqrt(g_y**2+g_x**2))
                theta[i, j] = np.rad2deg(np.arctan2(abs(g_y), g_x))
            
            else:
                G[i, j] = np.sqrt(g_y**2+g_x**2)
                theta[i, j] = np.rad2deg(np.arctan2(g_y, g_x))
                
    return G, theta

def _getHoGHist(G, theta, lr_steps, ud_steps, angleDict, num_bins, window, unsigned):
    
    if window != None:
        hist = np.zeros((ud_steps, num_bins*lr_steps))
    
        for i in range(ud_steps):
            for j in range(lr_steps):
                mag = G[i*window:(i+1)*window, j*window:(j+1)*window]
                ang = theta[i*window:(i+1)*window, j*window:(j+1)*window]
                for k in range(mag.shape[0]):
                    for l in range(mag.shape[1]):
                        angle = ang[k, l]
                        if unsigned and angle == 180:
                            angle = 0
                        elif not unsigned and angle == 180:
                            angle = -180
                        
                        angleBin = angleDict[angle]
                        hist[i, (j*num_bins)+angleBin] += mag[k, l]
                    
    else:
        hist = np.zeros(num_bins)
        
        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                angle = theta[i, j]
                if unsigned and angle == 180:
                    angle = 0
                elif not unsigned and angle == 180:
                    angle = -180
                    
                angleBin = angleDict[angle]
                hist[angleBin] += G[i, j]
                    
    return hist



    
    
    
    
    
    
    
    
    
    
    
    
    