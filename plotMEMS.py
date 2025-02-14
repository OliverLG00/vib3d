# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:24:58 2024

@author: Oliver
"""

import os
import numpy as np
import pandas as pd
# import open3d as o3d
from scipy import sparse as sp
from scipy.stats import binned_statistic_dd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# from numba import jit
from tqdm import tqdm
import pickle
from matplotlib import pyplot as plt
from matplotlib import patheffects as pe


textwidth = 17 #cm
linewidth = (textwidth - 0.6) / 2
fontsize = 9
fontfamily = 'Times New Roman'
cm = 1/2.54

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
        name = path + '/' + file
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
        name = path + '/' + file
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

# @jit(nopython=False)
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

# @jit(nopython=False)
def createANURB(x, y, knotvecx=None, knotvecy=None, segmentsx:int=1, segmentsy:int=1, degree:int=3, xlim:list=[0,0], ylim:list=[0,0], method:str=None): # create A-matrix for weight coefficients
    
    if np.any(knotvecx==None):
        if segmentsx < 1:
            segmentsx = 1
        
        if method == 'periodic':
            dt = np.unique(np.diff(np.sort(x))) / (np.max(x)-np.min(x))
            dt = np.min(dt[np.nonzero(dt)])
            knotvecx = np.r_[ -np.flip(np.arange(dt, degree*dt+dt, dt)), np.linspace(0, 1, segmentsx+1), 1+np.arange(dt, degree*dt+dt, dt) ]
        else:
            knotvecx = np.r_[ np.repeat(0, degree), np.linspace(0, 1, segmentsx+1), np.repeat(1, degree) ]
        xmin, xmax = x.min() + xlim[0], x.max() - xlim[1]
        x = (x - xmin) / (xmax - xmin)
    else:
        xmin, xmax = knotvecx[degree], knotvecx[-degree-1]
        knotvecx = (knotvecx - xmin) / (xmax - xmin)
        x = (x - xmin) / (xmax - xmin)
    
    if np.any(knotvecy==None):
        if segmentsy < 1:
            segmentsy = 1
        
        if method == 'periodic':
            dt = np.unique(np.diff(np.sort(y))) / (np.max(y)-np.min(y))
            dt = np.min(dt[np.nonzero(dt)])
            knotvecy = np.r_[ -np.flip(np.arange(dt, degree*dt+dt, dt)), np.linspace(0, 1, segmentsy+1), 1+np.arange(dt, degree*dt+dt, dt) ]
        else:
            knotvecy = np.r_[ np.repeat(0, degree), np.linspace(0, 1, segmentsy+1), np.repeat(1, degree) ]
        ymin, ymax = y.min() + ylim[0], y.max() - ylim[1]
        y = (y - ymin) / (ymax - ymin)
    else:
        ymin, ymax = knotvecy[degree], knotvecy[-degree-1]
        knotvecy = (knotvecy - ymin) / (ymax - ymin)
        y = (y - ymin) / (ymax - ymin)
        
    order = degree + 1
    N = []
    for n in range(0, len(knotvecx)-order):
        Nx = baseN(knotvecx, x, n, order, order)
        # N.append( np.array(list(map(lambda m: Nx * baseN(knotvecy, y, m, order, order),range(0, len(knotvecy)-order) ))) )
        for m in range(0, len(knotvecy)-order):
            Ny = baseN(knotvecy, y, m, order, order)
            N.append( Nx * Ny )
        
    knotvecx = knotvecx * (xmax - xmin) + xmin
    knotvecy = knotvecy * (ymax - ymin) + ymin
    # return np.array(N).T, knotvecx, knotvecy
    return sp.csc_matrix(N).T, knotvecx, knotvecy

def findspline(Xtrain, ytrain, Xval, yval, frequency, num_segmentsx=[1, 35], num_segmentsy=[1, 14], method=None):
    print(f'\nSearch for mean geometry and its respective best number of spline segments')
    mx, my = 8, 4 # mean and frequency optimization
    Atrain, knotx, knoty = createANURB(Xtrain['X'], Xtrain['Y'], segmentsx=mx, segmentsy=my, xlim=[5e-2, 5e-2], ylim=[5e-2, 5e-2], degree=3, method=None)
    Aval, knotx, knoty = createANURB(Xval['X'], Xval['Y'], knotvecx=knotx, knotvecy=knoty, degree=3, method=None)

    val = np.outer( np.float64(Xtrain.index), 2*np.pi*np.array(frequency) )
    A = sp.hstack([ Atrain, Atrain.multiply( np.cos(val) ), Atrain.multiply(np.sin(val) ) ]).tocsr()
    x = np.append( sp.linalg.inv(A.T.dot(A).tocsc()).dot( A.T.dot(ytrain) ), 2*np.pi*np.array(frequency) )
    
    for _ in tqdm(range(5)):
        val = np.outer( np.float64(Xtrain.index), x[-1] )
        a, b = Atrain.dot(x[Atrain.shape[1]:2*Atrain.shape[1]]), Atrain.dot(x[2*Atrain.shape[1]:-1])
        w = ytrain - (a * np.cos(val.flatten()) + b * np.sin(val.flatten()))
        A = sp.hstack([ Atrain, 
                        Atrain.multiply( np.cos(val) ), 
                        Atrain.multiply( np.sin(val) ),
                        np.reshape( np.float64(Xtrain.index) * ( b * np.cos(val.flatten()) - a * np.sin(val.flatten()) ), (-1,1)),
                        ]).tocsr()
        x += sp.linalg.inv(A.T.dot(A).tocsc()).dot( A.T.dot(w) )
    frequency = x[-1].copy()
    
    print(f'\nAfter optimization frequency: {frequency/(2*np.pi)} Hz')
    valtrain = np.outer( np.float64(Xtrain.index), frequency )
    valval = np.outer( np.float64(Xval.index), frequency )
    mae, segments = [], []
    for i in tqdm(range(num_segmentsx[0], num_segmentsx[1]+1)):
        # print(f'Main axis with {i} segments')
        for j in range(num_segmentsy[0], num_segmentsy[1]+1):
            segments.append( (i,j) )
        
            A, knotvec1, knotvec2 = createANURB(Xtrain['X'], Xtrain['Y'], segmentsx=i, segmentsy=j)
            A = sp.hstack([ Atrain, A.multiply( np.cos(valtrain) ), A.multiply(np.sin(valtrain) ) ])

            x = sp.linalg.inv(A.T.dot(A).tocsc()).dot( A.T.dot(ytrain) )
            metrain = np.mean( np.abs(ytrain - A.dot(x)) )
            
            A, knotvecx, knotvecy = createANURB(Xval['X'], Xval['Y'], knotvecx=knotvec1, knotvecy=knotvec2)
            A = sp.hstack([ Aval, A.multiply( np.cos(valval) ), A.multiply(np.sin(valval) ) ])
            meval = np.mean( np.abs(yval - A.dot(x)) )
            mae.append( (metrain, meval) )
    
    # segments, mse = np.array(segments), np.array(mse)
    # xx, yy = np.meshgrid( np.array(segments)[:,0], np.array(segments)[:,1], indexing='ij')
    
    # fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection':'3d'})
    # ax.plot_surface(xx, yy, np.array(mse)[:,1].reshape(xx.shape), cmap='viridis')
    
    # ax.plot(segments, mse, label=['train set', 'validation set'])
    # ax.grid(which='minor', axis='y')
    # ax.set_xlabel('number of segments'); ax.set_ylabel('MSE'); plt.legend()
    # ax.spines[['right', 'top']].set_visible(False)
    # fig.tight_layout()
    return mae, segments #segments[ np.array(mse)[:,1].argmin() ] # [:,1] takes the mse computed from validation

def findfrequency(Xtrain, ytrain, Xval, yval, frequencies, num_samples:int=1e6):
    print(f'\nSearch for the most dominant frequency:')
    num_freq = np.min([len(Xtrain), num_samples])
    idx = np.random.choice( np.arange(len(Xtrain)), int(num_samples), replace=False)
    Xtrain, ytrain = Xtrain.iloc[idx], ytrain.iloc[idx]
    
    time_train = 2*np.pi*np.float64(Xtrain.index)
    time_val = 2*np.pi*np.float64(Xval.index)
    mse, amp = [], []
    for f in tqdm(frequencies):
        valtrain = np.outer(time_train, f)
        valval = np.outer(time_val, f)
        A = np.c_[ np.cos(valtrain), np.sin(valtrain) ]
        xf = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ytrain) )
        msetrain = np.mean((ytrain - np.dot(A, xf))**2)
        
        A = np.c_[ np.cos(valval), np.sin(valval) ]
        mseval = 0 #np.mean((yval - np.dot(A, xf))**2)
        
        mse.append( (msetrain, mseval) )
        amp.append( np.linalg.norm(xf) )
    
    print(f'Found frequency at {frequencies[np.argmin(np.array(mse)[:,0])]} Hz')
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale/2), layout='constrained')
    ax.plot(frequencies, np.array(mse)[:,0] * 1e3) #label=['train set', 'validation set']
    ax.set_yscale('log')
    ax.set_xlabel('frequency [$Hz$]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_ylabel('MSE [$mm^2$]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.grid(which='minor', axis='y')
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    # ax.legend(prop={'size':fontsize*scale, 'family':fontfamily})
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale,linewidth*cm*scale/2), layout='constrained')
    ax.plot(frequencies, np.array(amp)**2 )
    ax.set_yscale('log')
    ax.set_xlabel('frequency [$Hz$]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_ylabel('PSD [Watt]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.grid(which='minor', axis='y')
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.spines[['right', 'top']].set_visible(False)
    fig.tight_layout()
    
    return frequencies[np.argmin(np.array(mse)[:,1])]


# def validateSpline(X, y, knotvecx, knotvecy, parameters):
#     Aspatial, knotx, knoty = createANURB(X['X'], X['Y'], knotvecx=knotvecx, knotvecy=knotvecy, degree=3, method=None)
#     val = np.outer(np.float64(X.index), 2*np.pi*parameters['frequency'])
#     mean = Aspatial.dot( parameters['mean geometry'] )
#     a = Aspatial.dot( parameters['Fourier variables'][0,::] )
#     b = Aspatial.dot( parameters['Fourier variables'][1,::] )
#     return y - ( mean + np.sum( a * np.cos(val) + b  * np.sin(val), axis=1) )

# def fourier2ampphas(x, y, knotvecx, knotvecy, parameters):
#     xx, yy = np.meshgrid( np.unique(x), np.unique(y), indexing='ij' )
#     A, knotx, knoty = createANURB(xx.flatten(), yy.flatten(), knotvecx=knotvecx, knotvecy=knotvecy, degree=3, method=None)
    
#     # amplitude = np.dot( A, np.sqrt( parameters['Fourier variables'][0,::]**2 + parameters['Fourier variables'][1,::]**2 ) )
#     # phase = np.dot( A, np.arctan2( parameters['Fourier variables'][1,::], parameters['Fourier variables'][0,::] ) )
#     amplitude = np.sqrt( A.dot(parameters['Fourier variables'][0,::])**2 + A.dot(parameters['Fourier variables'][1,::])**2 )
#     phase = np.arctan2( A.dot(parameters['Fourier variables'][1,::]), A.dot(parameters['Fourier variables'][0,::]) )
#     zza = np.stack(list(map(lambda i: amplitude[:,i].reshape(xx.shape), range(amplitude.shape[-1]) )), axis=-1 )
#     zzp = np.stack(list(map(lambda i: phase[:,i].reshape(xx.shape), range(phase.shape[-1]) )), axis=-1 )
#     return xx, yy, zza, zzp

def showresiduals(X, Xvec, Yvec, residual, clim=None, statistic='std', title=None ):
    factor = 1e3
    scale = 1
    
    xx, yy = np.meshgrid( np.unique(Xvec), np.unique(Yvec), indexing='ij' )
    zz = binned_statistic_dd(np.array(X[['X', 'Y']]), np.array(residual), statistic=statistic, bins=[np.unique(xx), np.unique(yy)]).statistic
    zz[ zz == 0 ] = np.nan # for counting
    
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, 1.5*linewidth*cm*scale))
    res = ax.imshow( np.abs(zz*factor), aspect='equal', origin='lower', extent=(X['Y'].min(), X['Y'].max(), X['X'].min(), X['X'].max()) )
    if clim != None:
        res.set_clim(0, clim*factor)
    cb = plt.colorbar(res)
    cb.ax.tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    cb.set_label('residual [mm]', fontsize=fontsize*scale, fontfamily=fontfamily)
    # ax.hlines(-0.06031936, X['Y'].min(), X['Y'].max(), 'r')
    ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel(r'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_ylabel(r'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    if title != None:
        ax.set_title(title, fontsize=fontsize*scale, fontfamily=fontfamily)
    fig.tight_layout()
    return xx, yy, zz
    
#%% Main
if __name__ == '__main__':
    #%% loading LiDAR DataFrames
    print(f'\nLivox loading . . .')
    ## only the plane
    xmin, xmax = 3.7, 4
    ymin, ymax = -0.76, 0.18
    zmin, zmax = -0.83, 1.08
    data = readLivox('./Livox', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=0)
    
    # print(f'\nBlickfeld loading . . .')
    # # only the plane
    # xmin, xmax = -0.47, 0.48
    # ymin, ymax = 3.7, 4
    # zmin, zmax = -0.7, 2
    # data = readBlickfeld('./Blickfeld', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=12.38709206)
    # source = o3d.geometry.PointCloud()
    # source.points = o3d.utility.Vector3dVector(data[['Y', 'X', 'Z']].to_numpy() * np.array([1, -1, 1]) )
    # source.paint_uniform_color([1, 0, 0])
    
    # # target = o3d.geometry.PointCloud()
    # # target.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].to_numpy() )
    # # target.paint_uniform_color([0, 0, 1])
    # source.points = o3d.utility.Vector3dVector( np.c_[ np.array(source.points)[:,0] - 0.05287782972943278, #np.array(target.points)[:,0].mean() - np.array(source.points)[:,0].mean(), 
    #                                                     np.array(source.points)[:,1],
    #                                                     np.array(source.points)[:,2] ] )
    # trans_init = np.array([[0.997361, 0.065720, 0.030847, 0.023127],
    #                         [-0.065361, 0.997783, -0.012493, -0.043477],
    #                         [-0.031600, 0.010443, 0.999446, -0.026073],
    #                         [0.000000, 0.000000, 0.000000, 1.000000]])
    # # o3d.visualization.draw_geometries([source.transform(trans_init), target])
    # data[['X', 'Y', 'Z']] = np.array( source.transform(trans_init).points )
    
    
    #%% applying PCA
    
    pca_transform = PCA()
    pca = data.copy()
    pca[['X', 'Y', 'Z']] = pca_transform.fit_transform(data[['X', 'Y', 'Z']])
    
    pca['Y'] = -pca['Y'] # Blickfeld with transform and Livox to properly align with the previous coordinate system
   
    #%% make orientation and pca axis plots
    scale = 2
    idx = np.random.choice(np.arange(data.shape[0]), 20000, replace=False)
    # colorsz = data['Z'].iloc[idx] < data['Z'].min() + (data['Z'].max() - data['Z'].min()) / 2
    # # colorsy = data['Y'].iloc[idx] < data['Y'].min() + (data['Y'].max() - data['Y'].min()) / 2 # Blickfeld with transform and Livox
    # colorsy = data['X'].iloc[idx] < data['X'].min() + (data['X'].max() - data['X'].min()) / 2 # Blickfeld without transform
    
    # _, ax = plt.subplots(1, 2, subplot_kw={'projection':'3d'}) # show orientation
    # ax[0].xaxis.pane.fill, ax[0].yaxis.pane.fill, ax[0].zaxis.pane.fill = False, False, False
    # ax[0].scatter(data['X'].iloc[idx][np.logical_and(colorsy, colorsz)], data['Y'].iloc[idx][np.logical_and(colorsy, colorsz)], data['Z'].iloc[idx][np.logical_and(colorsy, colorsz)], s=1)
    # ax[0].scatter(data['X'].iloc[idx][np.logical_and(colorsy, ~colorsz)], data['Y'].iloc[idx][np.logical_and(colorsy, ~colorsz)], data['Z'].iloc[idx][np.logical_and(colorsy, ~colorsz)], s=1)
    # ax[0].scatter(data['X'].iloc[idx][np.logical_and(~colorsy, colorsz)], data['Y'].iloc[idx][np.logical_and(~colorsy, colorsz)], data['Z'].iloc[idx][np.logical_and(~colorsy, colorsz)], s=1)
    # ax[0].scatter(data['X'].iloc[idx][np.logical_and(~colorsy, ~colorsz)], data['Y'].iloc[idx][np.logical_and(~colorsy, ~colorsz)], data['Z'].iloc[idx][np.logical_and(~colorsy, ~colorsz)], s=1)
    # ax[0].axis('equal')
    
    # ax[1].xaxis.pane.fill, ax[1].yaxis.pane.fill, ax[1].zaxis.pane.fill = False, False, False
    # ax[1].view_init(elev=30, azim=-240, roll=0) # default (30, -60, 0)
    # ax[1].scatter(pca['X'].iloc[idx][np.logical_and(colorsy, colorsz)], pca['Y'].iloc[idx][np.logical_and(colorsy, colorsz)], pca['Z'].iloc[idx][np.logical_and(colorsy, colorsz)], s=1)
    # ax[1].scatter(pca['X'].iloc[idx][np.logical_and(colorsy, ~colorsz)], pca['Y'].iloc[idx][np.logical_and(colorsy, ~colorsz)], pca['Z'].iloc[idx][np.logical_and(colorsy, ~colorsz)], s=1)
    # ax[1].scatter(pca['X'].iloc[idx][np.logical_and(~colorsy, colorsz)], pca['Y'].iloc[idx][np.logical_and(~colorsy, colorsz)], pca['Z'].iloc[idx][np.logical_and(~colorsy, colorsz)], s=1)
    # ax[1].scatter(pca['X'].iloc[idx][np.logical_and(~colorsy, ~colorsz)], pca['Y'].iloc[idx][np.logical_and(~colorsy, ~colorsz)], pca['Z'].iloc[idx][np.logical_and(~colorsy, ~colorsz)], s=1)
    # ax[1].axis('equal')
    # ax[1].set_ylim(ax[1].get_ylim()[::-1])
    # ax[1].set_zlim([-0.05, 0.05])
    
    mean = data[['X', 'Y', 'Z']].mean()
    ev = list(pca_transform.components_)
    pca_scale = 1.5
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale), subplot_kw={'projection':'3d'}, layout='tight')
    scat = ax.scatter( data['X'].iloc[idx], data['Y'].iloc[idx], data['Z'].iloc[idx], s=.1, c=data['Inc'].iloc[idx])
    ax.scatter( mean['X'], mean['Y'], mean['Z'], s=20, alpha=1, c='k')
    ax.quiver( mean['X'], mean['Y'], mean['Z'], ev[0][0], ev[0][1], ev[0][2], length=pca_scale, normalize=False, arrow_length_ratio=0.08, color='r')
    ax.quiver( mean['X'], mean['Y'], mean['Z'], ev[1][0], ev[1][1], ev[1][2], length=pca_scale, normalize=False, arrow_length_ratio=0.08, color='g')
    ax.quiver( mean['X'], mean['Y'], mean['Z'], -ev[2][0], -ev[2][1], -ev[2][2], length=pca_scale, normalize=False, arrow_length_ratio=0.08, color='b')
    
    ax.text(mean['X'] + ev[0][0]*pca_scale+0.005, mean['Y'] + ev[0][1]*pca_scale+0.032, mean['Z'] + ev[0][2]*pca_scale+0.01, r'$u$', fontsize=fontsize*scale, fontfamily=fontfamily, c='r', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
    ax.text(mean['X'] + ev[1][0]*pca_scale-0.1, mean['Y'] + ev[1][1]*pca_scale-0.032, mean['Z'] + ev[1][2]*pca_scale+0.01, r'$v$', fontsize=fontsize*scale, fontfamily=fontfamily, c='g', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
    ax.text(mean['X'] - ev[2][0]*pca_scale-0.01, mean['Y'] - ev[2][1]*pca_scale+0.032, mean['Z'] - ev[2][2]*pca_scale+0.01, r'$w$', fontsize=fontsize*scale, fontfamily=fontfamily, c='b', path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
    
    # cb = fig.colorbar(scat, cmap='viridis')
    # cb.set_label('intensity', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
    
    ax.axis('equal')
    ax.set_zlim([-1, 1])
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    # ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel('X [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
    ax.set_ylabel('Y [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
    ax.set_zlabel('Z [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
    
    #%% train test split
    # Xtrain, tmp = train_test_split(pca, train_size=0.7, shuffle=False)
    # Xval, Xtest = train_test_split(tmp, test_size=1/3, shuffle=False)
    # del data, tmp
    
    Xtrain = pca.iloc[ np.logical_and( pca.index > 13, pca.index < 80 ) ]
    Xval = pca.iloc[ np.logical_and( pca.index > 80, pca.index < 100) ]
    Xtest = pca.iloc[ np.logical_and( pca.index > 100, pca.index < 120 ) ]
    del data
    
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale/2), layout='constrained')
    ax.scatter(Xval.index, Xval['Z']*1e3, s=.1, alpha=.8)
    ax.set_xlim([80, 100])
    ax.set_ylim([-80, 80])
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel('time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=3*scale)
    ax.set_ylabel(f'$w$ [mm]',  fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=3*scale)
    #%% set hyperparameters
    mean_segx, mean_segy = 8, 4
    frequency = [0.249545, 0.49683] # 0.2479, 0.4979
    
    fsegments = [[35, 13], [23, 1]] # Livox harmonics: (27, 12), (34, 9) & Livox single: (35, 13), (23, 1)
    # fsegments = [[35, 14], [15, 4]] # Blickfeld harmonics: (34,14), (22,5) & Livox single: (35, 14), (15, 4)
    scale = 2
    
    Pbb = sp.eye(Xtrain.shape[0]) * 1
    Aspatial, mknotx, mknoty = createANURB(Xtrain['X'], Xtrain['Y'], segmentsx=mean_segx, segmentsy=mean_segy, degree=3, method=None)
    Aval, mknotx, mknoty = createANURB(Xval['X'], Xval['Y'], knotvecx=mknotx, knotvecy=mknoty, degree=3, method=None)
    #%% zero mean and search hyperparameters
    mean = sp.linalg.inv(Aspatial.T.dot(Pbb.dot(Aspatial))).dot(Aspatial.T.dot(Pbb).dot(Xtrain['Z']))
    zeromeantrain = Xtrain['Z'] - Aspatial.dot(mean)
    zeromeanval = Xval['Z'] - Aval.dot(mean)
    
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale/2), layout='constrained')
    ax.scatter(Xval.index, zeromeanval*1e3, s=.1, alpha=.8)
    ax.set_xlim([80, 100])
    ax.set_ylim([-80, 80])
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel('time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=3*scale)
    ax.set_ylabel(f'$w$ [mm]',  fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=3*scale)
    
    f = findfrequency(Xtrain[['X', 'Y']], zeromeantrain, Xval[['X', 'Y']], zeromeanval, np.arange(0.01, 20, 1e-4))
    # mae, segments = findspline(Xtrain[['X', 'Y']], Xtrain['Z'], Xval[['X', 'Y']], Xval['Z'], 0.24928, num_segmentsx=[1, 35], num_segmentsy=[1, 14])
    
    #%% compute A matrices for respective frequencies
    print(f'\nComputation works: {len(fsegments)==len(frequency)}!\n')
        
    Afreq, knot, nsplit = [], [[mknotx, mknoty]], [Aspatial.shape[1]]
    for seg in tqdm(fsegments):
        Af, knotx, knoty = createANURB(Xtrain['X'], Xtrain['Y'], segmentsx=seg[0], segmentsy=seg[1], xlim=[5e-2, 5e-2], ylim=[5e-2, 5e-2], degree=3, method=None)
        Afreq.append(Af)
        knot.append( [knotx.copy(), knoty.copy()] ); knot.append( [knotx.copy(), knoty.copy()] )
        nsplit.append(Af.shape[1]); nsplit.append(Af.shape[1])
    

    #%% adjustment
    print('\nOptimize given values:')
    def decx(x, knot, nsplit):
        x = np.split(x, np.cumsum(nsplit) )
        parameters = {'mean geometry': {'cv': x[0], 'knotvec': {'x': knot[0][0], 'y': knot[0][1]}},
                      'Fourier': [],
                      'frequency': x[-1]}
        for j in range(1, len(x)-1, 2):
            parameters['Fourier'].append( {'a': {'cv': x[j], 
                                                 'knotvec': {'x': knot[j][0], 
                                                             'y': knot[j][1]} }, 
                                           'b': {'cv':x[j+1], 
                                                 'knotvec': {'x': knot[j+1][0], 
                                                             'y': knot[j+1][1]} },
                                           } )
        return parameters
    
    # approximate values
    A = [Aspatial]
    for i, Af in enumerate(Afreq):
        val = np.outer( np.float64(Xtrain.index), 2*np.pi*frequency[i] )
        A.append( sp.hstack([ Af.multiply( np.cos(val) ), Af.multiply(np.sin(val) )]) )
    A = sp.hstack(A).tocsr()
    x = np.append( sp.linalg.inv(A.T.dot(Pbb).dot(A).tocsc()).dot(A.T.dot(Pbb).dot(Xtrain['Z'])), 2*np.pi*np.array(frequency) )
    parameters = decx(x, knot, nsplit )
    
    # adjustment and optimization
    for _ in tqdm(range(6)):
        A, F, fx = [Aspatial], [], Aspatial.dot(parameters['mean geometry']['cv'])
        for i, Af in enumerate(Afreq):
            val = np.outer( np.float64(Xtrain.index), parameters['frequency'][i] )
            a, b = Af.dot( parameters['Fourier'][i]['a']['cv'] ), Af.dot( parameters['Fourier'][i]['b']['cv'] )
            da, db = Af.multiply( np.cos(val) ), Af.multiply(np.sin(val) )
            df = sp.coo_matrix( np.float64(Xtrain.index) * ( b * np.cos(val.flatten()) - a * np.sin(val.flatten()) ) ).T
            A.append( sp.hstack([ da, db ]) )
            F.append(df)
            fx += a * np.cos(val.flatten()) + b * np.sin(val.flatten())
        A.append( sp.hstack(F) )
        A = sp.hstack( A ).tocsr()
        w = Xtrain['Z'] - fx
        dx = sp.linalg.inv(A.T.dot(Pbb).dot(A).tocsc()).dot(A.T.dot(Pbb).dot(w))
        print(f'  mean abs residual {np.round(np.mean(np.abs(w))*1e3, 4)} mm')
        x += dx
        parameters = decx(x, knot, nsplit)
    
    threshold = 0.02
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth/2*cm*scale))
    ax.scatter(Xtrain.index, w*1e3, s=.05, alpha=.6)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xlabel('time [sec]'); ax.set_ylabel(f'residual [mm]')
    ax.set_ylim([-threshold*1e3, threshold*1e3])
    fig.tight_layout()
    c = np.logical_or( w < -threshold, w > threshold )
    
    #%% saving dict
    # with open('saved_dictionary.pkl', 'wb') as f:
    #     pickle.dump(parameters, f)
        
    with open('domBlickfeld.pkl', 'rb') as f:
        parameters = pickle.load(f)
    #%% display
    # mx, my = np.meshgrid( np.arange(Xtrain['X'].min()+8e-2, Xtrain['X'].max()-5e-2, 1e-2), np.arange(Xtrain['Y'].min()+5e-2, Xtrain['Y'].max()-5e-2, 1e-2), indexing='ij' )
    mx, my = np.meshgrid( np.arange(-0.84, 0.919, 1e-2), np.arange(-0.42, 0.41, 4e-2), indexing='ij' )
    
    Afreq = []
    for seg in tqdm(parameters['Fourier']):
        Af, knotx, knoty = createANURB(mx.flatten(), my.flatten(), knotvecx=seg['a']['knotvec']['x'], knotvecy=seg['a']['knotvec']['y'], degree=3, method=None)
        Afreq.append(Af)
    
    for i, Af in tqdm(enumerate(Afreq)):
        a = Af.dot(parameters['Fourier'][i]['a']['cv']).reshape(mx.shape)
        b = Af.dot(parameters['Fourier'][i]['b']['cv']).reshape(mx.shape)
        
        amplitude = np.sqrt(a**2, b**2)
        print(f'Max amplitude {np.round(amplitude.max()*1e3, 2)} mm and mean amplitude of {np.round(amplitude.mean()*1e3, 2)} mm')
        phase = np.arctan2(b, a)
        
        # plot amplitude
        fig, ax = plt.subplots(figsize=(linewidth*cm*scale,linewidth*cm*scale*3/4), subplot_kw={'projection':'3d'}, layout='constrained')
        ax.view_init(elev=20, azim=-240, roll=0)
        ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
        ax.plot_wireframe(mx, my, amplitude, colors='k' ) # mesh
        # ax.scatter(Xtrain['X'][c], Xtrain['Y'][c], Xtrain['Z'][c], s=.05, alpha=.8) # "outliers
        
        ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)

        ax.set_xlabel(f'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=10*scale)
        ax.set_ylabel(f'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
        ax.set_zlabel('$w$ [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=2*scale)
        ax.set_box_aspect([2, 1, 0.75]) # ax.axis('equal')
        ax.set_zlim([0, 0.03])
        ax.set_xlim([-1, 1])
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_zticklabels([ str(int(float(l.get_text())*1e3)) for l in ax.get_zticklabels() ])
        # [ label.set_visible(False) for label in ax.get_xticklabels()[1::2]]
        [ label.set_visible(False) for label in ax.get_yticklabels()[::2]]
        [ label.set_visible(False) for label in ax.get_zticklabels()[1::2]]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        # plot phase
        fig, ax = plt.subplots(figsize=(linewidth*cm*scale,linewidth*cm*scale*3/4), subplot_kw={'projection':'3d'}, layout='constrained')
        ax.view_init(elev=20, azim=-240, roll=0)
        ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
        ax.plot_wireframe(mx, my, phase, colors='k' )
        
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)

        ax.set_xlabel(f'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=10*scale)
        ax.set_ylabel(f'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
        ax.set_zlabel('$w$ phase [rad]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=2*scale)

        # [ label.set_visible(False) for label in ax.get_xticklabels()[1::2]]
        [ label.set_visible(False) for label in ax.get_yticklabels()[::2]]
        ax.set_box_aspect([2, 1, 0.75]) # ax.axis('equal')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim( [ -np.pi, np.pi ] )
        # ax.set_ylim(ax.get_ylim()[::-1])
            
    
    