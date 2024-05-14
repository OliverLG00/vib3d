# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:36:08 2024

@author: Oliver
"""

import os
import numpy as np
import pandas as pd
import blickfeld_scanner as bfs

from jax import numpy as jnp
from jax import jit

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from scipy.signal import welch
from scipy.interpolate import interp1d
from scipy import sparse as sp
from scipy.stats import binned_statistic_dd

from numba import njit

import open3d as o3d
from matplotlib import pyplot as plt
from tqdm import tqdm
#%% parameters
np.set_printoptions(precision=5, linewidth=200)
textwidth = 7.16 # inch
linewidth = 3.5
cm = 1 # 1/2.54 when text_width unit = cm
scale = 2
xscale = 1e3 # xaxis value
yscale = 1e3 # yaxis value

figsize = (textwidth*cm*scale, linewidth*cm*scale)
textsize=10
fontfamily='Times New Roman'

local_textscale = scale / ( (linewidth*cm*scale) / (textwidth*cm*scale) ) # plot with textwidth for column
#%% functions
def polyA(X, poly_par):
    '''
    Create multi dimensional polynomial A matrix
    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    poly_par : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    '''
    mat = np.ones(len(X))
    for i, degree in enumerate(poly_par):
        mat = np.vstack(list(map(lambda d: mat * X[:,i]**d, reversed(range(int(degree)+1)) )))
    mat = mat.T
        
    # mat = ['']
    # for i, degree in enumerate(poly_par):
    #     tmp = []
    #     for j in reversed(range(int(degree)+1)):
    #         for k in mat:
    #             tmp.append( k + X[i] + str(j))
    #     mat = tmp.copy()
    return np.float64(mat)

#%%
if __name__ == '__main__':
#%% read profile
    # elemin, elemax = 256.9*np.pi/180, 284.5*np.pi/180
    # print(f'\nProfile loading . . .')
    # name = 'profile_static.txt'
    # data = pd.read_csv(name, sep=' ')
    # idx = np.logical_and( data['ele(rad)'] > elemin, data['ele(rad)'] < elemax )
    # data = data.drop(['r', 'g', 'b', 'add(decimal)', 'filterID'], axis='columns')
    # data = data[idx]
    # data['timestamp'] = data.index * 1/5e5 # point frequency
    
#%% show profile
    # _, ax = plt.subplots(figsize=figsize)
    # ax.scatter(data['y'], data['z'], s=.1, alpha=.8)
    
    # # ax.axis('equal')
    # # ax.set_xlim([-4.5, -3.5])
    # ax.xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    # ax.yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    # ax.set_xlabel('X', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    # ax.set_ylabel('Y', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    # ax.spines[['right', 'top']].set_visible(False)
        
#%% Blickfeld
    # path = '.\Blickfeld'
    # # whole experimental setup
    # # xmin, xmax = -0.7, 0.7
    # # ymin, ymax = 3.875, 4
    # # zmin, zmax = -2, 2
    
    # # only the plane
    # xmin, xmax = -0.47, 0.48
    # ymin, ymax = 3.7, 4
    # zmin, zmax = -0.7, 2
    
    # print(f'\nBlickfeld loading . . .')
    # data = []
    # for file in tqdm( os.listdir(path) ):
    #     name = path + '\\' + file
    #     tmp = pd.read_csv(name, sep=';')
    #     idx = np.logical_and( np.logical_and( np.logical_and( tmp['X'] < xmax, tmp['X'] > xmin ), 
    #                                         np.logical_and( tmp['Y'] < ymax, tmp['Y'] > ymin ) ),
    #                           np.logical_and( tmp['Z'] < zmax, tmp['Z'] > zmin ) ) # cut area of interest
    #     data.append( np.array(tmp[idx]) )
    # data = pd.DataFrame(np.vstack(data), columns=tmp.keys())
    # data['Time'] = (data['TIMESTAMP'] - data['TIMESTAMP'].min() ) * 1e-9
    # data.set_index('Time', inplace=True)
    # data['Inc'] = data['INTENSITY']
    # data['ID'] = data['POINT_ID']
    # data = data.drop(['DISTANCE', 'INTENSITY', 'POINT_ID', 'AMBIENT', 'RETURN_ID', 'TIMESTAMP'], axis='columns')
    
#%% Livox
    path = '.\Livox'
    ## whole experimental setup
    # xmin, xmax = 3.7, 4
    # ymin, ymax = -0.85, 0.275
    # zmin, zmax = -1.3, 1.20
    
    ## only the plane
    xmin, xmax = 3.7, 4
    ymin, ymax = -0.76, 0.18
    zmin, zmax = -0.83, 1.08
    
    print(f'\nLivox loading . . .')
    data = []
    for file in tqdm( os.listdir(path) ):
        name = path + '\\' + file
        tmp = pd.read_csv(name, sep=',')
        idx = np.logical_and( np.logical_and( np.logical_and( tmp['X'] < xmax, tmp['X'] > xmin ), 
                                            np.logical_and( tmp['Y'] < ymax, tmp['Y'] > ymin ) ),
                              np.logical_and( tmp['Z'] < zmax, tmp['Z'] > zmin ) ) # cut area of interest
        data.append( np.array(tmp[idx]) )
    data = pd.DataFrame(np.vstack(data), columns=tmp.keys())
    data['Time'] = (data['Timestamp'] - data['Timestamp'].min()) * 1e-9
    data.set_index('Time', inplace=True)
    data['Inc'] = data['Reflectivity']
    data = data.drop(['Reflectivity', 'Version', 'Slot ID', 'LiDAR Index', 'Rsvd', 'Error Code', 'Timestamp Type', 'Data Type', 'Timestamp', 'Tag', 'Ori_x', 'Ori_y', 'Ori_z', 'Ori_radius', 'Ori_theta', 'Ori_phi'], axis='columns')
    
#%% show 3D
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector( np.array(data[['X', 'Y', 'Z']]) )
    # o3d.visualization.draw_geometries([pc])
    
    # fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection':'3d'})
    # ax.scatter3D(data['X'], data['Y'], data['Z'], s=.1, alpha=.8, c=data['Inc'])
    
    # # ax.xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    # # ax.yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    # # ax.zaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
    # ax.set_xlabel('X', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    # ax.set_ylabel('Y', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    # ax.set_zlabel('Z', fontsize=textsize*local_textscale, fontfamily=fontfamily)
    
    # # ax.set_axis_off() # turn off everything
    # ax.xaxis.pane.fill = False
    # ax.yaxis.pane.fill = False
    # ax.zaxis.pane.fill = False
    
#%% mean geometry analysis
    pca_transform = PCA()
    pca = data.copy()
    pca[['X', 'Y', 'Z']] = pca_transform.fit_transform(data[['X', 'Y', 'Z']])

    Xtrain, tmp = train_test_split(pca, train_size=0.7, shuffle=False)
    Xval, Xtest = train_test_split(tmp, test_size=1/3, shuffle=False)
    
    # _, ax = plt.subplots()
    # ax.scatter(Xtrain.index, Xtrain['Z'], s=.1, alpha=.8)
    # ax.scatter(Xval.index, Xval['Z'], s=.1, alpha=.8)
    # ax.scatter(Xtest.index, Xtest['Z'], s=.1, alpha=.8)
    
    fmin, fmax = 1/(pca.index.max() - pca.index.min())/2, 1/np.mean(np.diff(pca.index.unique()))/2
    
#%% greedy geometry search full and batch wise
    dx, dy = np.meshgrid(np.arange(13, 18), np.arange(6, 10) )
    degree = np.c_[dx.flatten(), dy.flatten()]
    degree = list( np.r_[ np.c_[np.zeros(len(degree)), degree], np.c_[np.ones(len(degree)), degree] ] )
    
    X = np.array_split( np.c_[Xtrain.index, Xtrain[['X', 'Y', 'Z']]], np.arange(500000, len(Xtrain), 500000), axis=0)
    Xt = np.array_split( np.c_[Xtest.index, Xtest[['X', 'Y', 'Z']]], 5, axis=0)
    
    # losstrain, losstest = [], []
    # for d in tqdm(degree):
    #     A = polyA( np.c_[ Xtrain.index, Xtrain[['X', 'Y']] ], d)
    #     parameters = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, Xtrain['Z']))
    #     losstrain.append( np.mean( (Xtrain['Z'] - np.dot(A, parameters))**2 ) )
    #     losstest.append( np.mean( (Xtest['Z'] - np.dot(polyA(np.c_[ Xtest.index, Xtest[['X', 'Y']] ], d), parameters))**2 ) )
    
    # for i, d in enumerate(degree):
    #     print(f'Epoch {i+1} testing degree {d}')
    #     parameters, numel = [], []
    #     for batch in tqdm(X):
    #         numel.append(len(batch))
    #         A = polyA(batch[:,:-1], d)
    #         x = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T, batch[:,-1]))
    #         parameters.append(x.copy())
    #     x = np.average(parameters, weights=numel, axis=0)
    #     losstrain.append( np.sum( list(map(lambda batch: np.sum( (batch[:,-1] - np.dot(polyA(batch[:,:-1], d), x))**2 ), X )) ) / len(Xtrain) )
    #     losstest.append( np.sum( list(map(lambda batch: np.sum( (batch[:,-1] - np.dot(polyA(batch[:,:-1], d), x))**2 ), Xt)) ) / len(Xtest) )
            
    # best (0, 15, 8) (time, X, Y) batches and (0, 14, 8) full

#%% reduction of observations with the mean geometry
    dgeo = np.array([0, 15, 8]) # degree[np.argmin(losstestbatch)]
    
    parameters, numel = [], []
    for batch in tqdm(X):
        numel.append(len(batch))
        A = polyA(batch[:,:-1], dgeo)
        x = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T, batch[:,-1]))
        parameters.append(x.copy())
    x = np.average(parameters, weights=numel/np.sum(numel), axis=0)
    wtrain = np.float64(np.vstack( list(map(lambda batch: ( batch[:,-1] - np.dot(polyA(batch[:,:-1], d), x) ).reshape((-1,1)), tqdm(X) )) )).squeeze()
    wtest = np.float64(np.vstack( list(map(lambda batch: ( batch[:,-1] - np.dot(polyA(batch[:,:-1], d), x) ).reshape((-1,1)), tqdm(Xt) )) )).squeeze()
    

#%% greedy frequency search example
    freq = np.arange(fmin, 10, 1e-2)
    loss = []
    for f in tqdm(freq):
        val = np.float64( np.outer(Xtrain.index, 2*np.pi*f) )
        A = np.c_[ np.ones(len(Xtrain)), np.cos(val), np.sin(val) ]
        parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, wtrain) )
        loss.append( np.mean( (wtrain - np.dot(A, parameters))**2 ) )
    
    f = freq[np.argmin(loss)]
    results, numel = [], []
    for batch in tqdm(X):
        numel.append( len(numel) )
        Ageo = polyA( batch[:,:-1], dgeo )
        # if i == 0:
        val = np.float64( np.outer(batch[:,0], 2*np.pi*f) )
        A = np.c_[ Ageo, np.cos(val), np.sin(val) ]
        parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, batch[:,-1]))
        
        # else:
        #     val = np.float64( np.outer(batch[:,0], parameters[-1]) )
        #     time = np.float64(batch[:,0]).reshape((-1,1))
        #     A = np.c_[ Ageo, np.cos(val), np.sin(val), time * (parameters[-2] * np.cos(val) - parameters[-3] * np.sin(val)) ]
        #     parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, batch[:,-1] - np.dot(A[:,:-1], parameters[:-1]) ) )
        # print(f'%d iter angular velocity: %.3f' % (i+1, parameters[-1]) )
        results.append( parameters.copy() )
    x = np.float64( np.append( np.average(results, weights=numel/np.sum(numel), axis=0), 2*np.pi*f) )
    
    results, numel = [], []
    for _ in range(10):
        for batch in tqdm(X):
            numel.append(len(batch))
            time = np.float64(batch[:,0]).reshape((-1,1))
            val = np.outer(time, x[-1])
            A = np.c_[ polyA( batch[:,:-1], dgeo), np.cos(val), np.sin(val), time * (x[-2] * np.cos(val) - x[-3] * np.sin(val)) ]
            w = np.float64( batch[:,-1] - np.dot(A[:,:-1], x[:-1]) )
            dx = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, w ) )
            results.append( dx.copy() )
        dx = np.average(results, weights=numel/np.sum(numel), axis=0)
        print(f'%.5f' % (dx[-1]) )
        x += dx
            
#%% greedy Fourier variable (a and b) search
    dx, dy = np.meshgrid(np.arange(12, 15), np.arange(5, 9))
    degree = list( np.c_[ np.zeros(np.size(dx)), dx.flatten(), dy.flatten() ] )
    f = x[-1] # 1.569639062746273
    loss = []
    for i, d in enumerate(degree):
        print(f'Epoch {i+1} testing degree {d}')
        parameters, numel = [], []
        for batch in tqdm(X):
            numel.append(len(batch))
            Ageo = polyA( batch[:,:-1], dgeo)
            Aab = polyA( batch[:,:-1], d)
            val = np.float64( np.outer(batch[:,0], f) )
            A = np.c_[Ageo, Aab * np.cos(val), Aab * np.sin(val), ]
            x = np.dot( np.linalg.inv( np.dot(A.T, A) ), np.dot(A.T, batch[:,-1]))
            parameters.append( x.copy())
        x = np.float64( np.average(parameters, weights=numel/np.sum(numel), axis=0) )
        
        losstest = []
        for batch in Xt:
            Ageo = polyA(batch[:,:-1], dgeo)
            Aab = polyA(batch[:,:-1], d)
            val = np.float64(np.outer(batch[:,0], f))
            A = np.c_[ Ageo, Aab * np.cos(val), Aab * np.sin(val) ]
            losstest.append( np.sum( (batch[:,-1] - np.dot(A, x))**2 ) * (len(batch)/len(Xtest)) )
        loss.append( np.sum(losstest) )
    
#%% adjust for all previous
    dab = np.array([0, 14, 5])
    parameters, numel = [], []
    for batch in tqdm(X):
        numel.append(len(batch))
        time = np.float64(batch[:,0])
        val = np.outer(time, f)
        Ageo = polyA(batch[:,:-1], dgeo)
        Aab = polyA(batch[:,:-1], dab)
        A = np.c_[ Ageo, Aab * np.cos(val), Aab * np.sin(val) ]
        parameters.append( np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, batch[:,-1])) )
    x = np.append( np.average(parameters, weights=numel/np.sum(numel), axis=0), f)
    
    idx = np.array([Ageo.shape[1], Ageo.shape[1]+Aab.shape[1], Ageo.shape[1]+Aab.shape[1]*2])
    decx = lambda x: np.array_split(x, idx)
    
    for i in range(10):
        a = decx(np.float64(x))
        parameters, numel = [], []
        for batch in tqdm(X):
            numel.append(len(batch))
            time = np.float64(batch[:,0]).reshape((-1,1))
            val = np.outer(time, a[-1])
            Ageo = polyA(batch[:,:-1], dgeo)
            Aab = polyA(batch[:,:-1], dab)
            A = np.c_[ Ageo, Aab * np.cos(val), Aab * np.sin(val), time * (np.dot(Aab, a[2]).reshape((-1,1)) * np.cos(val) - np.dot(Aab, a[1]).reshape((-1,1)) * np.sin(val)) ]
            w = np.float64( batch[:,-1:] - (np.dot(Ageo, a[0]).reshape((-1,1)) + np.dot(Aab, a[1]).reshape((-1,1)) * np.cos(val) + np.dot(Aab, a[2]).reshape((-1,1)) * np.sin(val)) ).squeeze()
            parameters.append( np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, w)) )
        dx = np.average(parameters, weights=numel/np.sum(numel), axis=0)
        print(f'{i+1}: Maximum change {np.max(np.abs(dx))}')
        x += dx
    
    a, contra = decx(np.float64(x)), []
    for batch in tqdm(X):
        numel.append(len(batch))
        time = np.float64(batch[:,0]).reshape((-1,1))
        val = np.outer(time, a[-1])
        Ageo = polyA(batch[:,:-1], dgeo)
        Aab = polyA(batch[:,:-1], dab)
        A = np.c_[ Ageo, Aab * np.cos(val), Aab * np.sin(val), time * (np.dot(Aab, a[2]).reshape((-1,1)) * np.cos(val) - np.dot(Aab, a[1]).reshape((-1,1)) * np.sin(val)) ]
        contra.append( np.float64( batch[:,-1:] - (np.dot(Ageo, a[0]).reshape((-1,1)) + np.dot(Aab, a[1]).reshape((-1,1)) * np.cos(val) + np.dot(Aab, a[2]).reshape((-1,1)) * np.sin(val)) ).squeeze() )
    wtrain = np.array([])
    for w in contra: wtrain = np.append(wtrain, w)
#%% particle
    # particles = np.linspace(fmin, fmax, 20)
    # max_improvement = 3
    # threshold = 1e-5
    
    # @jit
    # def festf(Xtrain, ytrain, parameters=None):
    #     val = jnp.outer( Xtrain, parameters[-1] )
    #     a, b = parameters[1], parameters[2]
    #     A = jnp.c_[ jnp.ones(len(Xtrain)), jnp.cos(val), jnp.sin(val), Xtrain * (b * jnp.cos(val) - a * jnp.sin(val)) ]
    #     w = ytrain - jnp.dot(A[:,:-1], parameters[:-1])
    #     parameters += jnp.dot( jnp.linalg.inv(jnp.dot(A.T, A)), jnp.dot(A.T, w) )
    #     loss = jnp.mean( (ytrain - jnp.dot(A[:,:-1], parameters[:-1]))**2 )
    #     return parameters, loss
    
    # @jit
    # def fest(Xtrain, ytrain, frequency=0):
    #     val = jnp.outer( Xtrain, 2*jnp.pi*frequency )
    #     A = jnp.c_[ jnp.ones(len(Xtrain)), jnp.cos(val), jnp.sin(val) ]
    #     parameters = jnp.dot( jnp.linalg.inv(jnp.dot(A.T, A)), jnp.dot(A.T, ytrain) )
    #     w = ytrain - jnp.dot(A, parameters)
    #     loss = jnp.mean( w**2 )        
    #     return parameters,loss
    
    # def PSOf_evaluate(Xtrain, particles, max_improvement_steps=1, history=None):
    #     particle_freq, particle_loss = [], []
        
    #     if history == None:
    #         history = ([], [])
            
    #     # t = np.float64( Xtrain.index ).reshape((-1,1))
    #     for f in particles:
    #         # val = np.outer( t, 2*np.pi*f )
    #         # A = np.c_[ np.ones(len(Xtrain)), np.cos(val), np.sin(val) ]
    #         # parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, Xtrain['Z']) )
    #         # w = Xtrain['Z'] - np.dot(A, parameters)
    #         # loss = np.mean( w**2 )
    #         parameters, loss = fest(np.float64(Xtrain.index).reshape((-1,1)), np.float64(Xtrain['Z']), frequency=f)
    #         parameters = jnp.append(parameters, 2*np.pi*f)        
            
    #         for i in range(max_improvement):
    #             # val = np.outer( t, parameters[-1])
    #             # a, b = parameters[1], parameters[2]
    #             # A = np.c_[ np.ones(len(Xtrain)), np.cos(val), np.sin(val), t * (b * np.cos(val) - a * np.sin(val)) ]
    #             # dx = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, w) )
    #             # w = Xtrain['Z'] - np.dot(A[:,:-1], (parameters + dx)[:-1])
    #             # tmp = np.mean( w**2 )
    #             parameters_new, tmp = festf(np.float64(Xtrain.index).reshape((-1,1)), np.float64(Xtrain['Z']), parameters=parameters)
    #             if loss < tmp:
    #                 break
    #             else:
    #                 print(f' %.3f Hz at improvement %d loss %.8f' % (f, i+1, tmp) )
    #                 parameters = parameters_new.copy()
    #                 parameters.at[-1].set( jnp.clip( parameters[-1], 2*np.pi*fmin, 2*np.pi*fmax ) )
    #                 loss = tmp.copy()
    #         particle_freq.append( np.array(parameters[-1]) / (2*np.pi) )
    #         particle_loss.append( np.array(loss) )
    #     return np.array(particle_freq), np.array(particle_loss)
    
    # particle_freq, particle_loss = PSOf_evaluate(Xtrain, particles, 3)
    
    # ## initialize
    # particles = particle_freq.copy()
    # local_best = ( particle_freq.copy(),particle_loss.copy() )
    
    # global_best = ( particle_freq[np.argmin(particle_loss)], np.min(particle_loss) )
    # v = np.zeros(len(particles))
    
    # ## update
    # momentum, c1, c2 = 0.9, 2, 0.8
    
    # for i in tqdm(range(100)):
    #     v = momentum * v +\
    #         c1 * np.random.rand(len(particles)) * (local_best[0] - particles) +\
    #         c2 * np.random.rand(len(particles)) * (global_best[0] - particles)
    #     particles = np.clip(particles + v, fmin, fmax)
        
    #     particle_freq, particle_loss = PSOf_evaluate(Xtrain, particles, 3)
        
    #     idx = list(map(lambda j: (local_best[1][j] - particle_loss[j]) > 0, range(len(particles)) ))
        
    #     local_best[0][tmp] = particle_freq.copy()[tmp]
    #     local_best[1][tmp] = particle_loss.copy()[tmp]
        
    #     global_best = ( local_best[0][np.argmin(local_best[1])], np.min(local_best[1]) )
        
#%% 
    # xdegree = np.arange(1, 16)
    # ydegree = np.arange(1, 10)
    # tdegree = np.arange(1, 3)
    
    # X, Y, T = np.meshgrid(xdegree, ydegree, tdegree)
    
    # degree = list(np.c_[ X.reshape((-1,1)), Y.reshape((-1,1)), T.reshape((-1,1)) ])
    
    # d, batch_size = np.array([1, 1, 1]), (500, 1000)
    # batch_size = np.arange(batch_size[0], len(Xtrain)//2, batch_size[1])
    # parameters = []
    # for bs in tqdm( batch_size ):
    #     params, idx = [], np.arange(bs, len(Xtrain), bs)
    #     for batch, ybatch in zip(np.split( np.c_[Xtrain[['X', 'Y']], Xtrain.index].astype('float'), idx), np.split(np.array(Xtrain['Z']), idx) ):
    #         Atrain = polyA(batch, d)
    #         params.append( np.dot(np.linalg.inv(np.dot(Atrain.T, Atrain)), np.dot(Atrain.T, ybatch)) )
    #     parameters.append( [np.mean(params, axis=0), np.std(params, axis=0)] )
        
    # Atrain = polyA(np.c_[Xtrain[['X', 'Y']], Xtrain.index].astype('float'), d)
    # reference = np.dot(np.linalg.inv(np.dot(Atrain.T, Atrain)), np.dot(Atrain.T, Xtrain['Z']))
    
    # mean = np.array(list(map(lambda m: m[0], parameters)))
    # _, ax = plt.subplots()
    # ax.plot(batch_size, np.median(mean, axis=1))
    # ax.hlines(reference, 0, len(parameters))
    
    
    
        
    # losstrain, losstest = [], []
    # for d in tqdm(degree):
        # Atrain = polyA(np.c_[Xtrain[['X', 'Y']], Xtrain.index].astype('float'), d)
        # Atest = polyA(np.c_[Xtest[['X', 'Y']], Xtest.index].astype('float'), d)
        
        # parameters = np.dot(np.linalg.inv(np.dot(Atrain.T, Atrain)), np.dot(Atrain.T, Xtrain['Z']))
        
        # losstrain.append( np.mean((Xtrain['Z'] - np.dot(Atrain, parameters))**2) )
        # losstest.append( np.mean((Xtest['Z'] - np.dot(Atest, parameters))**2) )
    
    # plt.plot(losstrain)
    # plt.plot(losstest )
    
    #%%
    # fmax = 5
    # fparticles = np.linspace( fmin, fmax, 10 )
    
    # start_degree = [5, 1]
    # A1 = polyA( np.array(Xtrain[['X', 'Y']]), start_degree)
    # A2 = polyA( np.array(Xval[['X', 'Y']]), start_degree)
    
    # # PSO
    # degree = A1.shape[1]
    # mask_base = np.ones(A1.shape[1]); mask_base[-1] = 0

    # numf = 1
    # decx = lambda x: {'mean': x[:degree], 
    #                   'ab': x[degree:-numf].reshape((2, numf, degree)).swapaxes(1, 2), 
    #                   'frequency': x[-numf:] } # poly for amplitudes (a, b) x par x frequency
    
    # mask = np.append( np.tile(mask_base, 1+2*numf), np.zeros(numf) )
    
    
    # local_best, global_best = {'loss': [], 'frequency': []}, {'loss': np.inf, 'frequency': 0}
    # for f in tqdm(fparticles):
    #     time_train = np.float64( Xtrain['T'].to_numpy().reshape((-1,1)) )
    #     val_train = np.outer( time_train, 2*np.pi*f )
        
    #     time_val = np.float64( Xval['T'].to_numpy().reshape((-1,1)) )
    #     val_val = np.outer( time_val, 2*np.pi*f )
        
    #     A = sp.csr_matrix( np.c_[ A1, 
    #                              A1 * np.cos(val_train), 
    #                              A1 * np.sin(val_train),
    #                              ] )
    #     N = A.T.dot(A)
    #     x = sp.linalg.inv(N).dot(A.T.dot(Xtrain['Z']))
    #     w = Xtrain['Z'] - A.dot(x)
    #     tmp = np.mean( w**2 )
    #     x = np.append( x, f )
    #     parameters = decx( x )
        
    #     lam, R = np.zeros(len(x)), sp.diags( np.append( N.diagonal(), np.ones(numf) ) )
    #     while True:
    #         a = np.dot(A1, parameters['ab'][0,::])
    #         b = np.dot(A1, parameters['ab'][1,::])
    #         A = sp.csr_matrix( np.c_[ A1, 
    #                                   A1 * np.cos(val_train), 
    #                                   A1 * np.sin(val_train),
    #                                   time_train * (b * np.cos(val_train) - a * np.sin(val_train)),
    #                                   ] )
    #         N = A.T.dot(A)
    #         dx = sp.linalg.inv( N + R.multiply(lam) ).dot(A.T.dot(w)) + R.dot(x) * lam
    #         w = Xtrain['Z'] - A.dot(x + dx)
            
    #         if tmp > np.mean( w**2 ):                   
    #             a = np.dot(A2, parameters['ab'][0,::])
    #             b = np.dot(A2, parameters['ab'][1,::])
    #             Aval = sp.csr_matrix( np.c_[ A2, 
    #                                       A2 * np.cos(val_val), 
    #                                       A2 * np.sin(val_val),
    #                                       time_val * (b * np.cos(val_val) - a * np.sin(val_val)),
    #                                       ] )
    #             wlam = Aval.T.dot(Xval['Z'] - Aval.dot(x))  -  Aval.T.dot(Aval).dot(dx)
    #             Alam = sp.diags( R.dot(x) )
    #             # lam = sp.linalg.inv( Alam.T.dot(Alam) ).dot( Alam.T.dot(wlam) ) * mask
    #             lam = sp.linalg.spsolve( Alam.tocsr(), wlam ) * mask
    #             x += dx
    #             parameters = decx( x )
    #             tmp = np.mean( w**2 )
    #         else:
    #             break
        
    #     local_best['loss'].append( np.mean( (Xtrain['Z'] - np.dot(A.toarray(), x))**2 ) )
    #     local_best['frequency'].append( parameters['frequency'] )
    #     idx = np.argmin(local_best['loss'])
    #     global_best['loss'] = local_best['loss'][idx]
    #     global_best['frequency'] = local_best['frequency'][idx]
    
    
    #%%
    
    # deg = np.array([6, 1])
    # Atrain = polyA(np.array(Xtrain[['X', 'Y']]), deg)
    # Atest = polyA(np.array(Xtest[['X', 'Y']]), deg)
    
    # def greedy_frequency(Xtrain, Xtest, ytrain, ytest, df=1e-1):
    #     fmin, fmax = 1/(Xtrain.index.max() - Xtrain.index.min())/2, 1/np.mean(np.diff(Xtrain.index.unique()))/2
    #     freq = np.arange(fmin, 5, df)
        
    #     Atrain = polyA(np.array(Xtrain), deg)
    #     Atest = polyA(np.array(Xtest), deg)
        
    #     loss_train, loss_test = [], []
    #     for f in tqdm(freq):
    #         val = np.float64( np.outer(Xtrain.index, 2*np.pi*f) )
    #         A = np.c_[Atrain, Atrain * np.cos(val), Atrain * np.sin(val)]
    #         # parameters = sp.linalg.spsolve(A.T.dot(A), A.T.dot(ytrain) )
    #         parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ytrain) )
    #         # loss_train.append( np.mean( (ytrain - np.dot(A, parameters))**2) )

    #         val = np.float64( np.outer(Xtest.index, 2*np.pi*f) )
    #         A = sp.csr_matrix( np.c_[Atest, Atest * np.cos(val), Atest * np.sin(val)] )
    #         loss_test.append( np.mean( (ytest - A.dot( parameters ))**2) )
        
    #     f = freq[ np.argmin(loss_test) ]
    #     val = np.float64( np.outer(Xtrain.index, 2*np.pi*f) )
    #     A = np.c_[Atrain, Atrain * np.cos(val), Atrain * np.sin(val)]
    #     # parameters = sp.linalg.spsolve(A.T.dot(A), ytrain)
    #     parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ytrain) )
        
    #     return np.append(parameters, 2*np.pi*f), (freq, loss_test)
        
    # numf = 1
    # degree = Atrain.shape[1] # only for decx
    # decx = lambda x: {'mean': x[:degree], 
    #                   'ab': x[degree:-numf].reshape((2, numf, degree)).swapaxes(1, 2), 
    #                   'frequency': x[-numf:] } # poly for amplitudes (a, b) x par x frequency
    
    # x, loss = greedy_frequency(Xtrain[['X', 'Y']], Xtest[['X', 'Y']], Xtrain['Z'], Xtest['Z'], df=1e-2)
    

    # freq = np.arange(fmin, 1, 1e-3)
    
    # loss_train, loss_test = [], []
    # for f in tqdm(freq):
    #     val = np.float64( np.outer(Xtrain.index, 2*np.pi*f) )
    #     A = np.c_[Atrain, Atrain * np.cos(val), Atrain * np.sin(val)]
    #     parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, Xtrain['Z']) )
    #     loss_train.append( np.mean( (Xtrain['Z'] - np.dot(A, parameters))**2) )

    #     val = np.float64( np.outer(Xtest.index, 2*np.pi*f) )
    #     A = np.c_[Atest, Atest * np.cos(val), Atest * np.sin(val)]
    #     loss_test.append( np.mean( (Xtest['Z'] - np.dot(A, parameters))**2) )
    
    # _, ax = plt.subplots()
    # ax.plot(loss[0], loss[1])#; ax.plot(freq, loss_test)
    
    # loss = (freq, loss_test)
    # f = 0.2498 # loss[0][np.argmin(loss[1])] # min frequency
    # x = np.append(parameters, 2*np.pi*f)
    
    
   
    
    # epochs = 10
    # t = np.float64( Xtrain.index ).reshape((-1,1))
    # for i in range(epochs):
    #     parameters = decx( x )
    #     a = np.dot( Atrain, parameters['ab'][0,::] )
    #     b = np.dot( Atrain, parameters['ab'][1,::] )
    #     val = np.outer( t, parameters['frequency'] )
    #     A = sp.csr_matrix( np.c_[ Atrain, 
    #                               Atrain * np.cos(val), 
    #                               Atrain * np.sin(val),
    #                               t * (b * np.cos(val) - a * np.sin(val)) ])
    #     w = Xtrain['Z'] - A[:,:-len(parameters['frequency'])].dot(x[:-len(parameters['frequency'])])
    #     print(f' %d / %d loss: %f with frequency: %.5f' % (i+1, epochs, np.mean(w**2), parameters['frequency']/(2*np.pi)) )
    #     dx = sp.linalg.inv(A.T.dot(A)).dot(A.T.dot(w))
    #     # dx = 1e-6 * A.T.dot(w)
    #     x += dx
        
    # w = Xtrain['Z'] - A[:,:-len(parameters['frequency'])].dot(x[:-len(parameters['frequency'])])
    # parameters = decx( x )
    
    # _, ax = plt.subplots()
    # ax.scatter( Xtrain.index, Xtrain['Z'], s=.1, alpha=.8 )
    # ax.scatter( Xtrain.index, w, s=.1, alpha=.8 )
    # ax.set_xlim([0, 8])
    
    #%%
#     print(f'\nEstimate mean geometry . . .')
#     best = polynomial_PSO( np.array(pca[['X', 'Y']]), np.array(pca['Z']) )
    
#     # spec, dt = [], 0.42
#     # for i in tqdm(np.unique(pca['ID'])):
#     #     idx = pca['ID'] == i
#     #     pcat = pca['T'][idx]
#     #     pcaz = pca['Z'][idx]
#     #     t = np.arange(pcat.min(), pcat.max(), dt)
        
#     #     func = interp1d(pcat, pcaz, kind='cubic')
        
#     #     if pcat.shape[0] > 50:
#     #         [f, Pxx] = welch(func(t), nperseg=50, noverlap=50//2, detrend='linear')
#     #         spec.append( [f, Pxx] )
#     #     else:
#     #         spec.append( [[], []])
        
    
#     X, Y = np.meshgrid(np.arange( pca['X'].min(), pca['X'].max(), 5e-3 ), np.arange( pca['Y'].min(), pca['Y'].max(), 5e-3 ), indexing='ij')
#     x, y = X.flatten(), Y.flatten()
#     A = polyA(np.array(pca[['X', 'Y']]), best)
#     p = solve(A, np.array(pca['Z']))
#     z = np.dot( polyA(np.c_[x, y], best), p)
    
    
#     tmp = pca_transform.inverse_transform( np.c_[x, y, z] )
#     X, Y, Z = tmp[:,0].reshape(X.shape), tmp[:,1].reshape(X.shape), tmp[:,2].reshape(X.shape)
    
#     _, ax = plt.subplots(figsize=figsize, subplot_kw={'projection':'3d'})
#     # ax.scatter3D(pca['X'], pca['Y'], pca['Z'], s=.1, alpha=.8, c='tab:orange')
#     ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
#     ax.axis('equal')
#     ax.xaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
#     ax.yaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
#     ax.zaxis.set_tick_params(labelsize=textsize*local_textscale, labelfontfamily=fontfamily)
#     ax.set_xlabel('X', fontsize=textsize*local_textscale, fontfamily=fontfamily)
#     ax.set_ylabel('Y', fontsize=textsize*local_textscale, fontfamily=fontfamily)
#     ax.set_zlabel('Z', fontsize=textsize*local_textscale, fontfamily=fontfamily)
#     ax.xaxis.pane.fill = False
#     ax.yaxis.pane.fill = False
#     ax.zaxis.pane.fill = False

# #%% frequency analysis
#     fs = 2.3
#     duration = pca['T'].max() - pca['T'].min()
#     particles = np.sort( np.append( np.linspace(1/duration, fs, 10), 0.1) )
#     Af = polyA(np.array(pca[['X', 'Y']]), [1, 1])
    
#     val = np.outer(pca['T'], 2*np.pi*particles)
#     loss = []
#     for i, f in enumerate( list(val.T) ):
#         Atmp = np.c_[A, Af*np.cos(f.reshape((-1,1))), Af*np.sin(f.reshape((-1,1)))]
#         p = np.dot( np.linalg.inv(np.dot(Atmp.T, Atmp)), np.dot(Atmp.T, pca['Z']) )
#         loss.append( np.mean( (pca['Z'] - np.dot(Atmp, p))**2 ) ) # MSE
#         print(f'mean loss particle {i}: {loss[i]}')

#%% 
    # data = pd.read_csv('Livoxtmp.csv')
    # deg = np.array([1, 1, 6])
    # # A1 = polyA( ['x', 'y', 'z'], deg)
    # A1 = polyA( np.array(data[['X', 'Y', 'Z']]), deg )
    # # A1 = np.c_[ data[['X', 'Y', 'Z']], np.ones(len(data))]
    
    # parameters = np.random.rand(A1.shape[1]).reshape((-1,1))
    
    # w1 = np.dot(A1, parameters)
    # w2 = np.sum( parameters[:-1]**2 ) - 7
    
    # for i in range(50):
    #     A2 = np.append( 2*parameters[:-1], 0.).reshape((1,-1))
        
    #     N = np.r_[ np.c_[np.dot(A1.T, A1), -A2.T], np.append(-A2, 0.).reshape((1,-1)) ]
    #     Ni = np.linalg.inv(N)
        
    #     # du = np.dot( Ni, np.append( np.dot(A1.T, w1), w2 ) ).reshape((-1,1))
    #     a = np.dot( -Ni[:len(parameters), :len(parameters)], np.dot(A1.T, w1) )
    #     b = Ni[:len(parameters),-1:] * w2
    #     du = a + b
    #     parameters += du
        
    #     w1 = np.dot(A1, parameters)
    #     w2 = np.sum( parameters[:-1]**2 ) - 7
    #     print(f'loss: {np.mean(w1)}')
    
    
#%%
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split

# from tqdm import tqdm
# from matplotlib import pyplot as plt
# np.random.seed(42)

# t = np.arange(0, 10, 5e-7)

# freq = np.linspace(1/len(t)*10, np.round(1/np.median(np.diff(t))/2), 5, endpoint=False) # np.array([0.25, 120, 6e3, 4e4, 9e5])
# # a = np.array([3, 5, 2, 7, 0.5])
# # b = np.array([2, 1, 6, 10, 1])
# # a = (np.random.rand(len(freq))*2-1) * 4 
# # b = (np.random.rand(len(freq))*2-1) * 3
# a = np.ones(len(freq)) * 5
# b = np.ones(len(freq)) * 2
# noise = 1e-3

# val = np.outer(t, 2*np.pi*freq)
# y = pd.DataFrame(1 + np.sum(a * np.cos(val) + b * np.sin(val), axis=1) + (np.random.rand(len(t))*2-1)*noise, index=t, columns=['Z'])
# print(f'Generated time series with {noise} m noise')

# idx = np.sort( np.random.choice(np.arange(len(y)), int(5e6), replace=False) )
# t = t[idx]
# y = y.iloc[idx]

# _, ax = plt.subplots()
# ax.scatter(t, y, s=.1, alpha=.8)

# Xtrain, Xtest, ytrain, ytest = train_test_split(t, y, train_size=0.7, shuffle=False, random_state=42)
# Xtrain, Xtest, ytrain, ytest = np.array(Xtrain).squeeze(), np.array(Xtest).squeeze(), np.array(ytrain).squeeze(), np.array(ytest).squeeze()

# df = np.arange(0, 0.05, 1e-2)
# freqlist = []
# for f in freq:
#     print(f'\nfrequency test: {f}')
#     losslist = []
#     for shift in tqdm(df):
#         loss = []
#         val = np.outer( Xtrain, 2*np.pi* (f + shift) )
#         A = np.c_[ np.ones(len(ytrain)), np.cos(val), np.sin(val) ]
#         parameters = np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, ytrain) )
#         w = ytrain - np.dot(A, parameters)
#         loss.append( np.mean( w**2 ) )
#         parameters = np.append( parameters, 2*np.pi* (f + shift) )
        
#         for _ in range(10):
#             val = np.outer( Xtrain, parameters[-1] ).squeeze()
#             A = np.c_[ np.ones(len(ytrain)), np.cos(val), np.sin(val), Xtrain * (parameters[2] * np.cos(val) - parameters[1] * np.sin(val)) ]
#             parameters += np.dot( np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, w) )
#             w = ytrain - np.dot(A[:,:-1], parameters[:-1])
#             loss.append( np.mean( w**2 ) )
#         losslist.append(loss.copy())
#     freqlist.append(losslist.copy())
# # freqlist = np.array(freqlist)

# _, ax = plt.subplots()
# _ = list(map(lambda i: ax.plot(df, np.array(freqlist[i])[:,-1], label=f'{freq[i]} Hz'), range(len(freq)) ))
# ax.set_ylim([0, 120])
# # ax.set_xlim([0, ax.get_xlim()[1]])
# ax.legend()