# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:43:23 2024

@author: Oliver
"""

import os
import numpy as np
import pandas as pd
import open3d as o3d
from scipy import sparse as sp
from scipy.stats import binned_statistic_dd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
# from numba import jit
from tqdm import tqdm
from matplotlib import pyplot as plt

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

def findspline(Xtrain, ytrain, Xval, yval, num_segmentsx=[1, 5],num_segmentsy=[1, 5], method=None):
    print(f'\nSearch for mean geometry and its respective best number of spline segments')
    keystrain = Xtrain.keys()
    keysval = Xval.keys()
    
    mse, segments = [], []
    for i in tqdm(range(num_segmentsx[0], num_segmentsx[1]+1)):
        # print(f'Main axis with {i} segments')
        for j in range(num_segmentsy[0], num_segmentsy[1]+1):
            segments.append( (i,j) )
        
            A, knotvec1, knotvec2 = createANURB(Xtrain[keystrain[0]], Xtrain[keystrain[1]], segmentsx=i, segmentsy=j)
        
            x = sp.linalg.inv(A.T.dot(A).tocsc()).dot( A.T.dot(ytrain) )
            msetrain = np.mean( (ytrain - A.dot(x))**2 )
            
            A, knotvecx, knotvecy = createANURB(Xval[keysval[0]], Xval[keysval[1]], knotvecx=knotvec1, knotvecy=knotvec2)
            mseval = np.mean( (yval - A.dot(x))**2 )
            mse.append( (msetrain, mseval) )
            # mse.append( ( msetrain * 1e-2 * np.std(x), mseval * 1e-2 * np.std(x) ) )
            # mse.append( np.std(x) )
    
    # segments, mse = np.array(segments), np.array(mse)
    # xx, yy = np.meshgrid( np.array(segments)[:,0], np.array(segments)[:,1], indexing='ij')
    
    # fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection':'3d'})
    # ax.plot_surface(xx, yy, np.array(mse)[:,1].reshape(xx.shape), cmap='viridis')
    
    # ax.plot(segments, mse, label=['train set', 'validation set'])
    # ax.grid(which='minor', axis='y')
    # ax.set_xlabel('number of segments'); ax.set_ylabel('MSE'); plt.legend()
    # ax.spines[['right', 'top']].set_visible(False)
    # fig.tight_layout()
    return mse, segments #segments[ np.array(mse)[:,1].argmin() ] # [:,1] takes the mse computed from validation

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
        mseval = np.mean((yval - np.dot(A, xf))**2)
        
        mse.append( (msetrain, mseval) )
        amp.append( np.linalg.norm(xf) )
    
    print(f'Found frequency at {frequencies[np.argmin(np.array(mse)[:,1])]} Hz')
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale/2))
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
    
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale,linewidth*cm*scale/2))
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

### sparse version
def estimateSpline(X:pd.DataFrame, y:pd.DataFrame, frequencies:np.array, segmentsx:int=5, segmentsy:int=5, degree:int=3, xlim:list=[0,0], ylim:list=[0,0], num_iterations:int=5, viz:bool=False):
    Aspatial, knotx, knoty = createANURB(X['X'], X['Y'], segmentsx=segmentsx, segmentsy=segmentsy, degree=3, xlim=xlim, ylim=ylim, method=None)
    Aspatial = sp.csr_matrix(Aspatial)
    Pbb = sp.eye(Aspatial.shape[0]) * 1
    
    num_freq, spatial = len(frequencies), Aspatial.shape[1]
    decx = lambda x: {'mean geometry': x[:spatial], 
                      'Fourier variables': x[spatial:-num_freq].reshape((2, num_freq, spatial)).swapaxes(1, 2), 
                      'frequency': x[-num_freq:] }
    val = np.outer(np.float64(X.index), 2*np.pi*frequencies)
    A = sp.hstack( [Aspatial, 
              sp.hstack( list(map(lambda j: Aspatial.multiply( np.cos( val[:,j] ).reshape((-1,1)) ), range(val.shape[1]) )) ),
              sp.hstack( list(map(lambda j: Aspatial.multiply( np.sin( val[:,j] ).reshape((-1,1)) ), range(val.shape[1]) )) ), 
              ]).tocsr()
    x = np.append( sp.linalg.inv(A.T.dot(Pbb).dot(A)).tocsc().dot(A.T.dot(Pbb).dot(y)), 2*np.pi*frequencies )
    # x = np.append( np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, y)), 2*np.pi*frequencies )
    del A, val
    
    print(f'Optimizing {num_freq} frequencies:')
    mse = []
    for i in tqdm(range(num_iterations+1)):
        parameters = decx(x)
        val = np.outer( np.float64(X.index), parameters['frequency'] )
        a, b = Aspatial.dot( parameters['Fourier variables'][0,::] ), Aspatial.dot( parameters['Fourier variables'][1,::] )
        A = sp.hstack( [Aspatial, 
                  sp.hstack( list(map(lambda j: Aspatial.multiply( np.cos( val[:,j] ).reshape((-1,1)) ), range(val.shape[1]) )) ),
                  sp.hstack( list(map(lambda j: Aspatial.multiply( np.sin( val[:,j] ).reshape((-1,1)) ), range(val.shape[1]) )) ), 
                  np.float64(X.index).reshape((-1,1)) * (b * np.cos(val) - a * np.sin(val)),
                  ]).tocsr()
        
        w = y - A[:,:-num_freq].dot( x[:-num_freq] )
        mse.append( np.mean(w**2) )
        if i < num_iterations:
            x += sp.linalg.inv(A.T.dot(Pbb).dot(A).tocsc()).dot(A.T.dot(Pbb).dot(w))
        del A, a, b
    
    parameters = decx(x)
    parameters['frequency'] = parameters['frequency'] / (2*np.pi)
    if viz:
        ## MSE from observations
        fig, ax = plt.subplots(figsize=(linewidth*cm*scale,linewidth*cm*scale/2))
        ax.plot(mse)
        ax.set_yscale('log')
        ax.grid(which='minor', axis='y')
        ax.set_ylabel(f'MSE with {num_freq} frequencies', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax.set_xlabel('Iterations', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
    return parameters, knotx, knoty


def validateSpline(X, y, knotvecx, knotvecy, parameters):
    Aspatial, knotx, knoty = createANURB(X['X'], X['Y'], knotvecx=knotvecx, knotvecy=knotvecy, degree=3, method=None)
    val = np.outer(np.float64(X.index), 2*np.pi*parameters['frequency'])
    mean = Aspatial.dot( parameters['mean geometry'] )
    a = Aspatial.dot( parameters['Fourier variables'][0,::] )
    b = Aspatial.dot( parameters['Fourier variables'][1,::] )
    return y - ( mean + np.sum( a * np.cos(val) + b  * np.sin(val), axis=1) )

def fourier2ampphas(x, y, knotvecx, knotvecy, parameters):
    xx, yy = np.meshgrid( np.unique(x), np.unique(y), indexing='ij' )
    A, knotx, knoty = createANURB(xx.flatten(), yy.flatten(), knotvecx=knotvecx, knotvecy=knotvecy, degree=3, method=None)
    
    # amplitude = np.dot( A, np.sqrt( parameters['Fourier variables'][0,::]**2 + parameters['Fourier variables'][1,::]**2 ) )
    # phase = np.dot( A, np.arctan2( parameters['Fourier variables'][1,::], parameters['Fourier variables'][0,::] ) )
    amplitude = np.sqrt( A.dot(parameters['Fourier variables'][0,::])**2 + A.dot(parameters['Fourier variables'][1,::])**2 )
    phase = np.arctan2( A.dot(parameters['Fourier variables'][1,::]), A.dot(parameters['Fourier variables'][0,::]) )
    zza = np.stack(list(map(lambda i: amplitude[:,i].reshape(xx.shape), range(amplitude.shape[-1]) )), axis=-1 )
    zzp = np.stack(list(map(lambda i: phase[:,i].reshape(xx.shape), range(phase.shape[-1]) )), axis=-1 )
    return xx, yy, zza, zzp

def showresiduals(X, Xvec, Yvec, residual, clim=None, statistic='std', title=None ):
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
    data = readLivox('.\Livox', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=0)
    
    # print(f'\nBlickfeld loading . . .')
    # # only the plane
    # xmin, xmax = -0.47, 0.48
    # ymin, ymax = 3.7, 4
    # zmin, zmax = -0.7, 2
    # data = readBlickfeld('.\Blickfeld', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=12.38709206)
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
    
    #%% applying PCA and train test split
    
    pca_transform = PCA()
    pca = data.copy()
    pca[['X', 'Y', 'Z']] = pca_transform.fit_transform(data[['X', 'Y', 'Z']])

    # Xtrain, tmp = train_test_split(pca, train_size=0.7, shuffle=False)
    # Xval, Xtest = train_test_split(tmp, test_size=1/3, shuffle=False)
    # del data, tmp
    
    Xtrain = pca.iloc[ np.logical_and( pca.index > 13, pca.index < 80 ) ]
    Xval = pca.iloc[ np.logical_and( pca.index > 80, pca.index < 100) ]
    Xtest = pca.iloc[ np.logical_and( pca.index > 100, pca.index < 120 ) ]
    del data
    
    # zthreshold = 0.08
    # Xtrain = Xtrain[ Xtrain['Z'] < zthreshold ]
    # Xval = Xval[ Xval['Z'] < zthreshold ]
    # Xtest = Xtest[ Xtest['Z'] < zthreshold ]
    
    # minthreshold = np.float64(np.array(pca.min())[:2] + 0.04)
    # maxthreshold =  np.float64(np.array(pca.max())[:2] - 0.04)
    # Xtrain = Xtrain[ np.logical_and( np.logical_and( Xtrain['X'] > minthreshold[0], Xtrain['X'] < maxthreshold[0] ), 
    #                                   np.logical_and( Xtrain['Y'] > minthreshold[1], Xtrain['Y'] < maxthreshold[1]) ) ]
    # Xval = Xval[ np.logical_and( np.logical_and( Xval['X'] > minthreshold[0], Xval['X'] < maxthreshold[0] ), 
    #                               np.logical_and( Xval['Y'] > minthreshold[1], Xval['Y'] < maxthreshold[1]) ) ]
    # Xtest = Xtest[ np.logical_and( np.logical_and( Xtest['X'] > minthreshold[0], Xtest['X'] < maxthreshold[0] ), 
    #                               np.logical_and( Xtest['Y'] > minthreshold[1], Xtest['Y'] < maxthreshold[1]) ) ]
    
    ## Train test split
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.scatter(Xtrain.index, Xtrain['Z'], s=.1, alpha=.8, label='Train set')
    # ax.scatter(Xval.index, Xval['Z'], s=.1, alpha=.8, label='Validation set')
    # ax.scatter(Xtest.index, Xtest['Z'], s=.1, alpha=.8, label='Test set')
    # ax.spines[['top', 'right']].set_visible(False)
    # ax.set_xlabel('Time [sec]')
    # ax.set_ylabel('Movement [m]')
    # ax.legend()
    # fig.tight_layout()
    
    # step_width = 1e-2
    # Y = np.meshgrid( np.arange(pca['X'].min(), pca['Y'].max(), step_width), np.arange(pca['Y'].min(), pca['Y'].max(), step_width), indexing='xy')
    # X = np.c_[ Y[0].flatten(), Y[1].flatten() ]
    
    #%% mean spline
    # # mean_segments = 4 # # 4
    # mse, segments = findspline(Xtrain[['X', 'Y']], Xtrain['Z'], Xval[['X', 'Y']], Xval['Z'], num_segmentsx=[15, 50], num_segmentsy=[10, 25], method=None)
    
    # ## surface plot of cross validation 
    # mse, segments = np.array(mse), np.array(segments)
    # xx, yy = np.meshgrid(np.unique(segments[:,0]), np.unique(segments[:,1]), indexing='ij')
    # # mt, mv = mse[:,0].reshape(xx.shape), mse[:,1].reshape(xx.shape)
    # mt = mse.reshape(xx.shape)
    # fig, ax = plt.subplots(figsize=(10,10), subplot_kw={'projection':'3d'})
    # ax.plot_surface(xx, yy, mt, label='train')
    # # ax.plot_surface(xx, yy, mv, label='validation')
    # ax.set_xlabel('pca_0'); ax.set_ylabel('pca_1'); ax.set_zlabel('MSE'); ax.set_zscale('log')
    # fig.legend()
    
    ## plot raster of plane with derived pixelboundaries form the control points (knot vector)

    factor = 1e3
    # ## best found segments
    A, knotx, knoty = createANURB(Xtrain['X'], Xtrain['Y'], segmentsx=4, segmentsy=2, degree=3, method=None)
    x = sp.linalg.inv(A.T.dot(A).tocsc()).dot( A.T.dot( Xtrain['Z'] ) )
    A, knotx, knoty = createANURB(Xval['X'], Xval['Y'], knotvecx=knotx, knotvecy=knoty, degree=3, method=None)
    error = Xval['Z'] - A.dot(x)
    scale = 2
    _ = showresiduals(Xval, np.arange(Xval['X'].min(), Xval['X'].max(), 5e-2), np.arange(Xval['Y'].min(), Xval['Y'].max(), 5e-2), error, statistic='max' )

    # step = 1e-2
    # xx, yy = np.meshgrid( np.arange(knotx.min() + step, knotx.max()-step, step), np.arange(knoty.min() + step, knoty.max() - step, step), indexing='ij' )
    xx, yy = np.meshgrid( np.unique(knotx), np.unique(knoty), indexing='ij' )
    A, knotx, knoty = createANURB(xx.flatten(), yy.flatten(), knotvecx=knotx, knotvecy=knoty, degree=3, method=None)
    zz = A.dot(x).reshape(xx.shape)
    mean = pca_transform.inverse_transform( np.c_[ xx.flatten(), yy.flatten(), zz.flatten() ] )
    
    
    scale = 2
    fig, ax = plt.subplots(figsize=(linewidth*cm*scale, linewidth*cm*scale), subplot_kw={'projection':'3d'})
    # ax.scatter( a[:,0], a[:,1], a[:,2], alpha=.8)
    # ax.plot_wireframe(xx, yy, zz, colors='k'); ax.axis('equal'); ax.set_zlim([-0.01, 0.025])
    # ax.scatter( mean[:,0], mean[:,1], mean[:,2], alpha=.8)
    ax.plot_wireframe(mean[:,0].reshape(xx.shape), mean[:,1].reshape(xx.shape), mean[:,2].reshape(xx.shape), colors='k' ); ax.axis('equal'); #ax.set_xlim([3.85, 3.95]); ax.set_ylim([-1, 0.5])
    ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel('X [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale) # 4=default
    ax.set_ylabel('Y [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_zlabel('Z [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    fig.tight_layout()
    
    
    ## remaining error
    scale = 2
    ylim = 0.08 #np.max(np.abs( [np.min([Xval['Z'].min(), error.min()]), np.max([Xval['Z'].max(), error.max()])] ))
    t = Xval.index
    y = Xval['Z'] * factor
    fig, ax = plt.subplots( figsize=(linewidth*cm*scale,linewidth*cm*scale/2) )
    ax.scatter(t, Xval['Z'] * factor, s=.1, alpha=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel('Time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_ylabel('Movement [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim([-ylim*1e3, ylim*1e3])
    fig.tight_layout()
    
    fig, ax = plt.subplots( figsize=(linewidth*cm*scale, linewidth*cm*scale/2) )
    ax.scatter(t, error * factor, s=.1, alpha=.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax.set_xlabel('Time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_ylabel('Movement [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax.set_xlim([t.min(), t.max()])
    ax.set_ylim([-ylim*1e3, ylim*1e3])
    fig.tight_layout()

   #%% workflow
    segmentsx, segmentsy, frequency = 15, 12, [0.2479] #[0.2479, 0.4979, 2.4979]
    num_frequency, check_freq, num_iterations = 1, np.arange(1e-2, 20, 1e-2), 10
    num_samples = 1e5 # find frequency with n samples
    scale = 2
    
    if frequency == None:
        num_iter = num_frequency
    else:
        num_iter = len(frequency)
    
    ytrain, yval = Xtrain['Z'], Xval['Z']
    estimated_frequency = []
    for i in range(num_iter):
        if frequency == None:
            new_freq = findfrequency(Xtrain, ytrain, Xval, yval, check_freq, num_samples=num_samples)
            estimated_frequency.append( new_freq )
        else:
            estimated_frequency.append( frequency[i] )
        parameters, knotx, knoty = estimateSpline(Xtrain[['X', 'Y']], Xtrain['Z'], np.array(estimated_frequency), segmentsx=segmentsx, segmentsy=segmentsy, xlim=[5e-2, 5e-2], ylim=[5e-2, 5e-2], 
                                                  num_iterations=num_iterations, viz=True)
        ytrain = validateSpline(Xtrain[['X', 'Y']], Xtrain['Z'], knotx, knoty, parameters)
        yval = validateSpline(Xval[['X', 'Y']], Xval['Z'], knotx, knoty, parameters)
        
        ## as a timeseries remaining error
        fig, ax = plt.subplots( figsize=(linewidth*cm*scale, linewidth*cm*scale/2) )
        ax.scatter(t, yval*factor, s=.1, alpha=.8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.set_xlabel('Time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax.set_ylabel('Movement [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax.set_xlim([t.min(), t.max()])
        ax.set_ylim([-ylim*factor, ylim*factor])
        fig.tight_layout()
        
        estimated_frequency = list(parameters['frequency'])
        _ = showresiduals(Xval, np.arange(Xval['X'].min(), Xval['X'].max(), 5e-2), np.arange(Xval['Y'].min(), Xval['Y'].max(), 5e-2), yval, clim=ylim, statistic='max' )
    new_freq = findfrequency(Xtrain, ytrain, Xval, yval, check_freq, num_samples=num_samples)
    #%% plot
    step = 2.5e-2
    xx, yy, zza, zzp = fourier2ampphas(np.arange(knotx.min(), knotx.max(), step), np.arange(knoty.min(), knoty.max(), step), knotx, knoty, parameters)
    # xx, yy, zza, zzp = fourier2ampphas(np.unique(knotx), np.unique(knoty), knotx, knoty, parameters)
    
    scale = 2
    for i in range(zza.shape[-1]):
        fig, ax = plt.subplots(1, 2, figsize=(textwidth*cm*scale, linewidth*cm*scale), subplot_kw={'projection':'3d'})
        ax[0].plot_wireframe(xx, yy, zza[:,:,i], colors='k')
        ax[1].plot_wireframe(xx, yy, zzp[:,:,i], colors='k' )
        # ax[0].set_title('amplitude [mm]', fontsize=fontsize*scale, fontfamily=fontfamily)
        # ax[1].set_title('phase [rad]', fontsize=fontsize*scale, fontfamily=fontfamily)
        ax[0].set_xlabel(r'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax[0].set_ylabel(r'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax[0].set_zlabel(r'$w$ [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax[0].xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax[0].yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax[0].zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        
        ax[1].set_xlabel(r'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax[1].set_ylabel(r'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax[1].set_zlabel(r'$w$ [rad]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
        ax[1].xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax[1].yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax[1].zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        
        ax[0].xaxis.pane.fill, ax[0].yaxis.pane.fill, ax[0].zaxis.pane.fill = False, False, False
        ax[1].xaxis.pane.fill, ax[1].yaxis.pane.fill, ax[1].zaxis.pane.fill = False, False, False
        
        ax[0].axis('equal')
        ax[0].set_zlim( [ np.sum(zza, axis=-1).min(), np.sum(zza, axis=-1).max() ] )
        ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_ylim(ax[0].get_ylim()); ax[1].set_zlim( [ -np.pi, np.pi ] )
        fig.tight_layout()
        
        ax[0].set_zticklabels([ str(float(l.get_text())*factor) for l in ax[0].get_zticklabels() ])
    
    #%% dominant frequency
    # un = np.unique(np.diff(np.sort(pca.index)))
    # fmin, fmax = 1/(pca.index.max() - pca.index.min()), 20# 1/np.min(un[un>0])
    # frequencies = np.arange(fmin, fmax, 0.01)
    
    # # ## compute the mean geometry again with the best mse
    # Atrain, knotx, knoty = createANURB(Xtrain['X'], Xtrain['Y'], knotvecx=knotx, knotvecy=knoty, degree=3, method=None)
    # Aval, knotx, knoty = createANURB(Xval['X'], Xval['Y'], knotvecx=knotx, knotvecy=knoty, degree=3, method=None)
    
    # x = np.dot( np.linalg.inv(np.dot(Atrain.T, Atrain)), np.dot(Atrain.T, Xtrain['Z']) )
    # ytrain = Xtrain['Z'] - np.dot(Atrain, x)
    # yval = Xval['Z'] - np.dot(Aval, x)
    
    # dominant_freq = findfrequency(Xtrain, ytrain, Xval, yval, frequencies)
    dominant_freq = 0.2479
    
   
    
    #%% second frequency
    # ## remaining error
    # y = yval * 1e3
    # fig, ax = plt.subplots( figsize=(linewidth*cm*scale, linewidth*cm*scale/2) )
    # ax.scatter(t, y, s=.1, alpha=.8)
    # ax.spines[['top', 'right']].set_visible(False)
    # ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    # ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    # ax.set_xlabel('Time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    # ax.set_ylabel('Movement [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    # ax.set_xlim([t.min(), t.max()])
    # ax.set_ylim([-ylim*1e3, ylim*1e3])
    # fig.tight_layout()
    
    # second_freq = findfrequency(Xtrain, ytrain, Xval, yval, frequencies)
    second_freq = 0.4979
    
    #%% third frequency
    # ## remaining error
    # y = yval * 1e3
    # fig, ax = plt.subplots( figsize=(linewidth*cm*scale, linewidth*cm*scale/2) )
    # ax.scatter(t, y, s=.1, alpha=.8)
    # ax.spines[['top', 'right']].set_visible(False)
    # ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    # ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    # ax.set_xlabel('Time [sec]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    # ax.set_ylabel('Movement [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    # ax.set_xlim([t.min(), t.max()])
    # ax.set_ylim([-ylim*1e3, ylim*1e3])
    # fig.tight_layout()
    
    # third_freq = findfrequency(Xtrain, ytrain, Xval, yval, frequencies)
    third_freq = 2.4979
    
    #%% full plot
    xx, yy = np.meshgrid( np.arange(knotx.min() + step, knotx.max()-step, step), np.arange(knoty.min() + step, knoty.max() - step, step), indexing='ij' )
    # xx, yy = np.meshgrid( np.unique(knotx), np.unique(knoty), indexing='ij' )
    A, knotx, knoty = createANURB(xx.flatten(), yy.flatten(), knotvecx=knotx, knotvecy=knoty, degree=3, method=None)
    mean = A.dot(parameters['mean geometry']).reshape(xx.shape)
    if mean[ xx.shape[0]//2, xx.shape[1]//2 ] < 0:
        mirror = -1
    else:
        mirror = 1
    shift = np.min(mean * mirror)
    
    amplitude = np.sqrt( A.dot(parameters['Fourier variables'][0,::])**2 + A.dot(parameters['Fourier variables'][1,::])**2 )
    # _ = [ mean.append(amplitude[:,i].reshape(xx.shape)) for i in range(amplitude.shape[1]) ]
    amplitude = list(map(lambda i: amplitude[:,i].reshape(xx.shape), range(amplitude.shape[1]) ))
    amplitude = list(np.cumsum(np.stack(amplitude), axis=0))
    
    phase = np.arctan2( A.dot(parameters['Fourier variables'][1,::]), A.dot(parameters['Fourier variables'][0,::]) )
    
    # geo = [pca_transform.inverse_transform( np.c_[ xx.flatten(), yy.flatten(), mean.flatten() ] ).reshape(tuple(np.append(xx.shape, 3) ) )]
    # geo = list(map(lambda i: pca_transform.inverse_transform( np.c_[ xx.flatten(), yy.flatten(), amplitude[:,:,i].flatten() ] ).reshape(tuple(np.append(xx.shape, 3) ) ))
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    labels = list(map(lambda i: f'amplitude {i+1}', range(len(amplitude)) )); labels.insert(0, 'mean')
    
    fig, ax = plt.subplots(1, 2, figsize=(textwidth*cm*scale, linewidth*cm*scale), subplot_kw={'projection':'3d'})
    ax[0].plot_wireframe(xx, yy, mean*mirror - shift, colors='k')
    
    _ = [ ax[0].plot_wireframe(xx, yy, mean*mirror - shift + a, colors=colors[i%len(colors)]) for i, a in enumerate(amplitude) ]
    # ax[0].plot_wireframe(xx, yy, mean*factor, colors='k')
    ax[0].set_xlabel(r'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax[0].set_ylabel(r'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax[0].set_zlabel(r'$w$ [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax[0].xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax[0].yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax[0].zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax[0].legend(labels, prop={'size':fontsize*scale, 'family':fontfamily})
    ax[0].axis('equal')
    ax[0].set_zlim( [ 0, np.max(mean*mirror - shift + amplitude[-1]) ] )
    ax[0].view_init(30, -60)
    ax[0].set_zticklabels([ str(float(l.get_text())*factor) for l in ax[0].get_zticklabels() ])
    
    
    xyz = pca_transform.inverse_transform( np.c_[ xx.flatten(), yy.flatten(), mean.flatten() ] ).reshape(tuple(np.append(xx.shape, 3)))
    ax[1].plot_wireframe(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], colors='k')
    
    for i, a in enumerate(amplitude):
        xyz = pca_transform.inverse_transform( np.c_[ xx.flatten(), yy.flatten(), (mean + a*mirror).flatten() ] ).reshape(tuple(np.append(xx.shape, 3)))
        ax[1].plot_wireframe(xyz[:,:,0], xyz[:,:,1], xyz[:,:,2], colors=colors[i%len(colors)])
    ax[1].set_xlabel('X [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax[1].set_ylabel('Y [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax[1].set_zlabel('Z [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=4*scale)
    ax[1].xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax[1].yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax[1].zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
    ax[1].axis('equal')
    ax[1].set_xlim([3.8, 3.95])
    ax[1].set_ylim([-0.6, 0.6])
    ax[1].legend(labels, prop={'size':fontsize*scale, 'family':fontfamily})
    ax[1].view_init(30, 115)
    
    ax[0].xaxis.pane.fill, ax[0].yaxis.pane.fill, ax[0].zaxis.pane.fill = False, False, False
    ax[1].xaxis.pane.fill, ax[1].yaxis.pane.fill, ax[1].zaxis.pane.fill = False, False, False
    
    #%% Cross validation
    num_freq = len(parameters['frequency'])
    
    ytrain = validateSpline(Xtrain[['X', 'Y']], Xtrain['Z'], knotx, knoty, parameters)
    yval = validateSpline(Xval[['X', 'Y']], Xval['Z'], knotx, knoty, parameters)
    ytest = validateSpline(Xtest[['X', 'Y']], Xtest['Z'], knotx, knoty, parameters)
    
    print(f'cross-validation: \n MSE_train: {np.mean(ytrain**2)} \n MSE_val: {np.mean(yval**2)} \n MSE_test: {np.mean(ytest**2)}')
