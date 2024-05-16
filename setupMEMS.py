# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:43:23 2024

@author: Oliver
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from matplotlib import pyplot as plt

#%%
def readLivox(path, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], zlim=[-np.inf, np.inf], timeoffset=0.):
    '''
    Read Livox LiDAR frames to a DataFrame with keys ['X', 'Y', 'Z', 'Inc'].
    Parameters
    ----------
    path : string
        The folder where Livox frames are stored.
    xlim : list, optional
        Spatial limitation along the x-axis. The default is [-np.inf, np.inf].
    ylim : list, optional
        Spatial limitation along the y-axis. The default is [-np.inf, np.inf].
    zlim : list, optional
        Spatial limitation along the z-axis. The default is [-np.inf, np.inf].
    timeoffset : float, optional
        Shift in time given in seconds. The default is 0.

    Returns
    -------
    data : DataFrame
        Data with cartesian coordinates and the Intensity as keys and timestamps as index.

    '''
    data = []
    for file in tqdm( os.listdir(path) ):
        name = path + '\\' + file
        tmp = pd.read_csv(name, sep=',')
        idx = np.logical_and( np.logical_and( np.logical_and( tmp['X'] < xlim[1], tmp['X'] > xlim[0] ), 
                                                np.logical_and( tmp['Y'] < ylim[1], tmp['Y'] > ylim[0] ) ),
                                  np.logical_and( tmp['Z'] < zlim[1], tmp['Z'] > zlim[0] ) ) # cut area of interest
        data.append( np.array(tmp[idx]) )
    data = pd.DataFrame(np.vstack(data), columns=tmp.keys())
    data['Time'] = (data['Timestamp'] - data['Timestamp'].min()) * 1e-9 + timeoffset# change milliseconds to seconds and add an offset in time
    data.set_index('Time', inplace=True)
    data['Inc'] = data['Reflectivity']
    data = data.drop(data.keys()[list(np.sum(np.vstack(list(map(lambda s: data.keys() == s, ['X', 'Y', 'Z', 'Inc'])) ), axis=0) == 0)], axis='columns')
    return data

def readBlickfeld(path, xlim=[-np.inf, np.inf], ylim=[-np.inf, np.inf], zlim=[-np.inf, np.inf], timeoffset=0.):
    '''
    Read Blickfeld LiDAR frames to a DataFrame with keys ['X', 'Y', 'Z', 'Inc'].
    Parameters
    ----------
    path : string
        The folder where Blickfeld frames are stored.
    xlim : list, optional
        Spatial limitation along the x-axis. The default is [-np.inf, np.inf].
    ylim : list, optional
        Spatial limitation along the y-axis. The default is [-np.inf, np.inf].
    zlim : list, optional
        Spatial limitation along the z-axis. The default is [-np.inf, np.inf].
    timeoffset : float, optional
        Shift in time given in seconds. The default is 0.

    Returns
    -------
    data : DataFrame
        Data with cartesian coordinates and the Intensity as keys and timestamps as index.

    '''
    data = []
    for file in tqdm( os.listdir(path) ):
        name = path + '\\' + file
        tmp = pd.read_csv(name, sep=';')
        idx = np.logical_and( np.logical_and( np.logical_and( tmp['X'] < xlim[1], tmp['X'] > xlim[0] ), 
                                                np.logical_and( tmp['Y'] < ylim[1], tmp['Y'] > ylim[0] ) ),
                                  np.logical_and( tmp['Z'] < zlim[1], tmp['Z'] > zlim[0] ) ) # cut area of interest
        data.append( np.array(tmp[idx]) )
    data = pd.DataFrame(np.vstack(data), columns=tmp.keys())
    data['Time'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min() ) * 1e-9 + timeoffset # change milliseconds to seconds and add an offset in time
    data.set_index('Time', inplace=True)
    data['Inc'] = data['INTENSITY']
    data['ID'] = data['POINT_ID']
    data = data.drop(data.keys()[list(np.sum(np.vstack(list(map(lambda s: data.keys() == s, ['X', 'Y', 'Z', 'Inc'])) ), axis=0) == 0)], axis='columns')
    return data

def baseN(knotvec, t, i, n, order): # compute basis for segments depending on t. i are knotvec indices
    """
    Parameters
    ----------
    knotvec : numpy array
        knotvector represents the segments (range 0-1)
    t : numpy array
        x-axis value (range 0-1)
    i : int
        should run from range( 0, (len(knotvec) - order) ) as input
    n : int
        stays order all the time (lookup)
    order : int
        variable for iteration
    
    Returns
    -------
    
    """
    if order-1 == 0:
        temp1 = np.logical_and( knotvec[i] <= t, t < knotvec[i + 1] )
        ## account for rounding issues original
        # temp2 = np.logical_and( i == 0, t < knotvec[n-1])
        # temp3 = np.logical_and( i == len(knotvec) - n-1, knotvec[-n] <= t )
        
        ## account for rounding issues (last 0 and first 1 are targeted)
        # temp2 = np.logical_and( i == 0, t < knotvec[n-1])
        # temp3 = np.logical_and( i == len(knotvec)-n, knotvec[-n] <= t )
        
        ## account for rounding issues (i = 0 & degree-1; knotvec first and last value of real segments!) (virtual knotvec values are nor considered)
        temp2 = np.logical_and( i == 0, t < knotvec[n-1])
        temp3 = np.logical_and( i== len(knotvec)-n-1, knotvec[-n] <= t )
        
        N = np.logical_or( temp1, np.logical_or( temp2, temp3) ).astype(float) # 0 or 1
    else:
        denom1 = knotvec[i + order-1] - knotvec[i] ## alpha_i **(n-1)
        denom2 = knotvec[i + order] - knotvec[i + 1] ## alpha_(i+1) **(n-1)
        
        term1 = 0.
        term2 = 0.
        if denom1 != 0:
            term1 = (t - knotvec[i]) / denom1  *  baseN(knotvec, t, i, n, order-1)
        if denom2 != 0:
            # term2 = (1 - (t - knotvec[i+1])/denom2) * baseN(knotvec, t, i+1, order-1) # original
            term2 = (knotvec[i+order] - t) / denom2  *  baseN(knotvec, t, i+1, n, order-1) #rearanged
        N = term1 + term2
    return N

def createAspline(x, knotvec=None, segments=0, degree=3, method=None): # create A-matrix for weight coefficients
    
    if np.any(knotvec==None):
        if segments < 1:
            segments = 1
        
        if method == 'periodic':
            dt = np.min(np.diff(x)) / (np.max(x)-np.min(x))
            knotvec = np.r_[ -np.flip(np.arange(dt, degree*dt+dt, dt)), np.linspace(0, 1, segments+1), 1+np.arange(dt, degree*dt+dt, dt) ]
        else:
            knotvec = np.r_[ np.repeat(0, degree), np.linspace(0, 1, segments+1), np.repeat(1, degree) ]
        xmin, xmax = x.min(), x.max()
        x = (x - xmin) / (xmax - xmin)
    else:
        xmin, xmax = knotvec.min(), knotvec.max()
        knotvec = (knotvec - xmin) / (xmax - xmin)
        x = (x - xmin) / (xmax - xmin)
    
    order = degree + 1
    N = []
    for m in range(0, len(knotvec)-order):
        N.append( baseN(knotvec, x, m, order, order) )
    knotvec = knotvec * (xmax - xmin) + xmin
    return np.array(N).T, knotvec

def estimatespline(time, data, frequency, segments=1, degree=3, method=None, max_iter=100):
    """
    Parameters
    ----------
    time : Series/ numpy array
        time vector for observations in data.
    data : numpy array [nx2]
        observed coordinates. First column object direction and second column oscillation direction. numpy.c_[ Y, Z ]
    frequency : list or vector
        starting frequency values.
    segments : int, optional
        how many spline segments are used to approximate the object. The default is 1. (equally spaces objects here)
    degree : int, optional
        spline degree. The default is 3.
    method : string, optional
        when set to "periodic" spaces first and last value of the control vector equally. The default is None.
    max_iter : int, optional
        max number of iterations. The default is 100.

    Returns
    -------
    list
        [0] mean,
        [1] [a&b, par, frequency]
        [2] frequency.
    Covariance matrix of unknown
    history of unknown vector

    """
    time = np.array(time); data = np.array(data)
    y = np.arange(np.min(data[:,0]), np.max(data[:,0]), 1e-3)
    decx = lambda x: [ x[:degree+segments], x[degree+segments:-len(frequency)].reshape((2, len(frequency), degree+segments)).swapaxes(1, 2), x[-len(frequency):] ] # poly for amplitudes (a, b) x par x frequency
    
    val = np.outer(time, 2*np.pi*frequency)

    A_spline, knotvec = createAspline(data[:,0], segments=segments, degree=degree, method=method ) # corresponding weights to the observations
    
    A = np.c_[ A_spline, 
              np.concatenate(list(map(lambda i: A_spline * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
              np.concatenate(list(map(lambda i: A_spline * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1), 
              ]
    # A = sp.csr_matrix(A)
    # x = sp.linalg.inv( A.T.dot(A) ).dot( A.T.dot(data[:,1]) )
    x = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, data[:,1]) )
    x = np.append(x, 2*np.pi*frequency)
    
    max_stat = binned_statistic_dd( data[:,0], data[:,1], 'max', bins=[y] )
    min_stat = binned_statistic_dd( data[:,0], data[:,1], 'min', bins=[y] )
    
    a = np.stack(list(map(lambda i: np.dot(A_spline, decx(x)[1][0, :,i]), range(len(frequency)) ))).T # [y.shape, len(frequency)]
    b = np.stack(list(map(lambda i: np.dot(A_spline, decx(x)[1][1, :,i]), range(len(frequency)) ))).T
    
    multiplier = np.nanmax((max_stat[0] - min_stat[0])/2) / np.max( np.sum(np.sqrt(a**2 + b**2), axis=1) )
    x[degree+segments:-len(frequency)] = x[degree+segments:-len(frequency)] * multiplier
    
    history = []
    Pbb = sp.eye(data.shape[0]) * 1/(3.6e-3**2)  #Leica P50 accuracy
    for _ in tqdm( range(max_iter) ):
        unknown = decx(x)
        val = np.outer(time, unknown[-1])
        a = np.stack( list(map(lambda i: np.dot(A_spline, unknown[1][0, :, i]), range(len(unknown[-1])) )) ).T
        b = np.stack( list(map(lambda i: np.dot(A_spline, unknown[1][1, :, i]), range(len(unknown[-1])) )) ).T
        A = np.c_[ A_spline, 
                  np.concatenate(list(map(lambda i: A_spline * np.cos(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  np.concatenate(list(map(lambda i: A_spline * np.sin(val[:,i]).reshape((-1,1)), range(val.shape[1]) )), axis=1),
                  time.reshape((-1,1)) * (b*np.cos(val) - a*np.sin(val)),
                  ]
        A = sp.csr_matrix(A)
        w = data[:,1] - A[:,:-len(frequency)].dot( x[:-len(frequency)] )
        N = sp.linalg.inv(A.T.dot(Pbb).dot(A) ).tocsc()
        x += N.dot( A.T.dot(Pbb).dot(w) )
        # w = data[:,1] - np.dot(A[:,:-len(frequency)], x[:-len(frequency)])
        # N = np.linalg.inv( np.dot(A.T, A) )
        # x += np.dot( N, np.dot(A.T, w) )
        history.append(np.copy(x))
    history = np.array(history)
    return decx(x), N.toarray(), knotvec, np.asarray(history)

#%% Main
if __name__ == '__main__':
    #%% loading LiDAR DataFrames
    print(f'\nLivox loading . . .')
    ## only the plane
    xmin, xmax = 3.7, 4
    ymin, ymax = -0.76, 0.18
    zmin, zmax = -0.83, 1.08
    data = readLivox('.\Livox', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=0)
    
    # print(f'\nBlickfeld loading . . .')
    # ## only the plane
    # xmin, xmax = -0.47, 0.48
    # ymin, ymax = 3.7, 4
    # zmin, zmax = -0.7, 2
    # data = readBlickfeld('.\Blickfeld', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=0)
    
    #%% applying PCA and train test split
    
    pca_transform = PCA()
    pca = data.copy()
    pca[['X', 'Y', 'Z']] = pca_transform.fit_transform(data[['X', 'Y', 'Z']])

    Xtrain, tmp = train_test_split(pca, train_size=0.7, shuffle=False)
    Xval, Xtest = train_test_split(tmp, test_size=1/3, shuffle=False)
    
    #%% mean spline
    # print(f'Search for mean geometry and its respective best number of spline segments')
    # mse, segments = [], []
    # for i in tqdm(range(1, 10+1)):
    #     segments.append( i )
    #     Ax, knotvec_x = createAspline(Xtrain['X'], segments=i) # spline in the pca x-axis
    #     Ay, knotvec_y = createAspline(Xtrain['Y'], segments=i) # spline in the pca y-axis
    #     A = Ax * Ay
        
    #     x = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, Xtrain['Z']) )
    #     msetrain = np.mean((Xtrain['Z'] - np.dot(A, x))**2)
        
    #     Ax, knotvec_x = createAspline(Xval['X'], segments=i) # spline in the pca x-axis
    #     Ay, knotvec_y = createAspline(Xval['Y'], segments=i) # spline in the pca y-axis
    #     A = Ax * Ay
    #     mseval = np.mean((Xval['Z'] - np.dot(A, x))**2)
    #     mse.append( (msetrain, mseval) )
    
    mean_segments = 4 #segments[ np.array(mse)[:,1].argmin() ] # 4
    
    #%%
    print(f'Search for the first most dominant frequency:')
    un = np.unique(np.diff(np.sort(pca.index)))
    fmin, fmax = 1/(pca.index.max() - pca.index.min()), 20# 1/np.min(un[un>0])
    frequencies = np.arange(fmin, fmax, 0.01)
    
    Ax, knotvec_x = createAspline(Xtrain['X'], segments=mean_segments) # spline in the pca x-axis
    Ay, knotvec_y = createAspline(Xtrain['Y'], segments=mean_segments) # spline in the pca y-axis
    Atrain = Ax * Ay
    x = np.dot( np.linalg.inv(np.dot(Atrain.T, Atrain)), np.dot(Atrain.T, Xtrain['Z']) )
    ytrain = Xtrain['Z'] - np.dot(Atrain, x)
    
    Ax, knotvec_x = createAspline(Xval['X'], knotvec=knotvec_x) # spline in the pca x-axis
    Ay, knotvec_y = createAspline(Xval['Y'], knotvec=knotvec_y) # spline in the pca y-axis
    Aval = Ax * Ay
    yval = Xval['Z'] - np.dot(Aval, x)
    
    time_train = 2*np.pi*np.float64(Xtrain.index)
    time_val = 2*np.pi*np.float64(Xval.index)
    mse = []
    for f in tqdm(frequencies):
        valtrain = np.outer(time_train, f)
        valval = np.outer(time_val, f)
        A = np.c_[ np.cos(valtrain), np.sin(valtrain) ]
        xf = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ytrain) )
        msetrain = np.mean((ytrain - np.dot(A, xf))**2)
        
        A = np.c_[ np.cos(valval), np.sin(valval) ]
        mseval = np.mean((yval - np.dot(A, xf))**2)
        
        mse.append( (msetrain, mseval) )
    
    dominant_freq = frequencies[np.argmin(np.array(mse)[:,1])] # 0.2479 Hz