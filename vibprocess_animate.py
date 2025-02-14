# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:08:24 2024

@author: Oliver
"""

import os
import numpy as np
import pandas as pd
import pickle
import blickfeld_scanner as bfs
from sklearn.decomposition import PCA

import open3d as o3d
from time import time
from scipy import sparse as sp
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
from matplotlib import animation
from tqdm import tqdm
#%% settings
np.set_printoptions(precision=5, linewidth=200)
textwidth = 7.16 # inch
linewidth = 3.5
cm = 1 # 1/2.54 when text_width unit = cm
scale = 2
xscale = 1e3 # xaxis value
yscale = 1e3 # yaxis value

figsize = (textwidth*cm*scale, linewidth*cm*scale)
textsize=10
# linewidth=(textsize-0.6)/2
fontfamily='Times New Roman'
fontsize=9

local_textscale = scale / ( (linewidth*cm*scale) / (textwidth*cm*scale) ) # plot with textwidth for column

#%% functions
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

def ani_mov(i):
    alpha_decrease = 1e2
    t0 = 0
    if i <= duration:
        cutout = data.iloc[(data.index - t0) < (i+1) * delay]
    else:
        cutout = data.iloc[np.logical_and( (data.index - t0) >= (i-duration) * delay, (data.index - t0) < (i+1) * delay )]
    # cutout = data.iloc[:i][-50000:]
    # a = np.linspace(0, 0.8, cutout.shape[0])
    a = np.float64( 0.8 * alpha_decrease**(cutout.index - cutout.index.max() ) )
    
    sc._offsets3d = (cutout['X'], cutout['Y'], cutout['Z'])
    sc._A = np.float64(cutout['Inc'])
    sc._alpha = a
    # ax.clear()
    # # ax.scatter( cutout['Y'], cutout['Z'], s=.1, alpha=.8, c=cutout['X'] )
    
    # ax.scatter(cutout['X'], cutout['Y'], cutout['Z'], s=.1, alpha=a, c=cutout['Inc'])
    # ax.axis('equal');    ax.set_axis_off()
    # ax.set_xlim([0, 20]);   ax.set_ylim([-2,2]);    ax.set_zlim([-1.5, 1.5])
    # ax.set_xlabel('X [m]');    ax.set_ylabel('Y [m]');    ax.set_zlabel('Z [m]')
    
def ani_mov_wf(i):
    ## point observation
    alpha_decrease = 1e2
    t0 = 0
    if i <= duration:
        cutout = data.iloc[(data.index - t0) < (i+1) * delay]
    else:
        cutout = data.iloc[np.logical_and( (data.index - t0) >= (i-duration) * delay, (data.index - t0) < (i+1) * delay )]
    # cutout = data.iloc[:i][-50000:]
    # a = np.linspace(0, 0.8, cutout.shape[0])
    a = np.float64( 0.8 * alpha_decrease**(cutout.index - cutout.index.max() ) )
    
    sc._offsets3d = (cutout['X'], cutout['Y'], cutout['Z'])
    sc._A = np.float64(cutout['Inc'])
    sc._alpha = a
    
    ## wireframe
    global wf
    if wf:
        wf.remove()
    
    # move = np.zeros(mx.shape)
    move = mean.copy()
    for j, m in enumerate(modes):
        move += m[0] * np.cos(parameters['frequency'][j]*i*delay) + m[1] * np.sin(parameters['frequency'][j]*i*delay) 
    
    # j, move = 0, np.zeros(mx.shape)
    # move += modes[j][0] * np.cos(parameters['frequency'][j]*i*delay) + modes[j][1] * np.sin(parameters['frequency'][j]*i*delay) 
    
    wf = ax.plot_wireframe(mx, my, move, colors='k')
    
#%%
if __name__ == '__main__':
    #%% Blickfeld
        path = '.\Blickfeld'
        # whole experimental setup
        # xmin, xmax = -0.7, 0.7
        # ymin, ymax = 3.875, 4
        # zmin, zmax = -2, 2
        
        # only the plane
        xmin, xmax = -0.47, 0.48
        ymin, ymax = 3.7, 4
        zmin, zmax = -0.7, 2
        
        print(f'\nBlickfeld loading . . .')
        data = readBlickfeld('./Blickfeld', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=12.38709206)
        data_b = data.copy()
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
        data = readLivox('./Livox', xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax], timeoffset=0)

    #%% registration
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(data_b[['Y', 'X', 'Z']].to_numpy() * np.array([1, -1, 1]) )
        source.paint_uniform_color([1, 0, 0])
        
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].to_numpy() )
        target.paint_uniform_color([0, 0, 1])
        # # o3d.visualization.draw_geometries([source, target])
        
        # threshold = 0.01  # [[cR, T], [0, 0, 0, 1]]
        # # trans_init = np.array([[1, 0, 0, 0],
        # #                        [0, 1, 0, 0],
        # #                        [0, 0, 1, 0],
        # #                        [0, 0, 0, 1]])
        trans_init = np.array([[0.997361, 0.065720, 0.030847, 0.023127],
                                [-0.065361, 0.997783, -0.012493, -0.043477],
                                [-0.031600, 0.010443, 0.999446, -0.026073],
                                [0.000000, 0.000000, 0.000000, 1.000000]])
        # # reg_p2p = o3d.pipelines.registration.registration_icp(
        # #             source, target, threshold, trans_init,
        # #             o3d.pipelines.registration.TransformationEstimationPointToPoint(with_scaling=False),
        # #             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50, relative_fitness=1e-7),
        # #             )
        
        # # params = o3d.geometry.KDTreeSearchParamRadius(radius=0.1)
        # # source.estimate_normals(params); target.estimate_normals(params)
        # # reg_p2p = o3d.pipelines.registration.registration_icp(
        # #             source, target, threshold, trans_init,
        # #             o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        # #             o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
        # #             )
        
        # # print(reg_p2p.transformation)
        source.points = o3d.utility.Vector3dVector( np.c_[ np.array(source.points)[:,0] - 0.05287782972943278, #np.array(target.points)[:,0].mean() - np.array(source.points)[:,0].mean(), 
                                                            np.array(source.points)[:,1],
                                                            np.array(source.points)[:,2] ] )
        # o3d.visualization.draw_geometries([source.transform(trans_init), target])
        data_b[['X', 'Y', 'Z']] = np.array(source.points)

    #%% time shift
    
        # from sklearn.decomposition import PCA
        # pca_transform = PCA()
        # liv = pca_transform.fit_transform( data[['X', 'Y', 'Z']] - data[['X', 'Y', 'Z']].mean(axis=0) )
        # bli = pca_transform.transform( data_b[['X', 'Y', 'Z']] - data_b[['X', 'Y', 'Z']].mean(axis=0) )
        # limitt = 120 # + 0.9345 works well & - 6.4990409200000006
        # corr = np.correlate( liv[np.logical_and(data.index < limitt, liv[:,-1] <= 0.08),-1], bli[data_b.index < limitt,-1], mode='same' )
        # shift = data.index[ np.argmax( corr ) ]
        
        # fig, ax = plt.subplots(2, 1)
        # ax[0].plot(data.index[np.logical_and(data.index<limitt, liv[:,-1]<=0.08)], corr)
        # ax[1].scatter( data.index, liv[:,-1], s=.1, alpha=.8 )
        # ax[1].scatter( data_b.index + shift, bli[:,-1], s=.1, alpha=.8 )
        # # plt.scatter( data.index, liv[:,-1], s=.1, alpha=.8 )
        # # plt.scatter( data_b.index - shift, bli[:,-1], s=.1, alpha=.8 )
        
        # shift = -6.4990409200000006
        # data_b.index = data_b.index + shift
        # data = data_b.copy()
        
    #%% pca
        pca_transform = PCA()
        data[['X', 'Y', 'Z']] = pca_transform.fit_transform(data[['X', 'Y', 'Z']])
        # data['Y'] = -data['Y'] # Blickfeld with transform and Livox to properly align with the previous coordinate system
        
    #%% load parameters
        with open('domLivox.pkl', 'rb') as f:
            parameters = pickle.load(f)
        step = 5e-2
        binx = np.arange(parameters['Fourier'][0]['a']['knotvec']['x'].min(), parameters['Fourier'][0]['a']['knotvec']['x'].max(), step)
        biny = np.arange(parameters['Fourier'][0]['a']['knotvec']['y'].min(), parameters['Fourier'][0]['a']['knotvec']['y'].max(), step)
        mx, my = np.meshgrid(binx.min() + step/2 + np.cumsum(np.diff(binx)), 
                              biny.min() + step/2 + np.cumsum(np.diff(biny)), indexing='ij')

        # mean = createANURB(mx.flatten(), my.flatten(), parameters['mean geometry']['knotvec']['x'], parameters['mean geometry']['knotvec']['y'], degree=3, method=None)[0].dot(parameters['mean geometry']['cv']).reshape(mx.shape)
        # modes = []
        # for p in parameters['Fourier']:
        #     modes.append(( createANURB(mx.flatten(), my.flatten(), p['a']['knotvec']['x'], p['a']['knotvec']['y'], degree=3, method=None)[0].dot(p['a']['cv']).reshape(mx.shape),
        #             createANURB(mx.flatten(), my.flatten(), p['b']['knotvec']['x'], p['b']['knotvec']['y'], degree=3, method=None)[0].dot(p['b']['cv']).reshape(mx.shape) ))
        
        with open('single_Livox.pkl', 'rb') as f:
            parameters = pickle.load(f)
            
        mean = griddata(parameters['coordinates'], parameters['mean geometry'], (mx.flatten(), my.flatten()), method='cubic').reshape(mx.shape)
        modes = []
        for p in parameters['Fourier']:
            modes.append(( griddata(parameters['coordinates'], p['a'], (mx.flatten(), my.flatten()), method='cubic').reshape(mx.shape),
                          griddata(parameters['coordinates'], p['b'], (mx.flatten(), my.flatten()), method='cubic').reshape(mx.shape) ))
        
    #%%
        delay = ( data.index.max() - data.index.min() ) / len(data) # time between frames in seconds
        fps = 30
        delay = 1/fps
        duration = 6
        frames = int(np.round( ( data.index.max() - data.index.min() ) / delay )) # for entire time series
        # frames = int(np.round( 1/(parameters['frequency'][0]/(2*np.pi)) / delay )*2 ) # for specific frequency
        scale = 2
        # idx = data.index <= 0.2
        
        fig, ax = plt.subplots(figsize=(15,15), subplot_kw={'projection':'3d'})
        ax.view_init(elev=20, azim=-240, roll=0)
        sc = ax.scatter( [], [], [], s=.1, alpha=.8, c=[])
        wf = None
        # wf = ax.plot_wireframe(mx, my, mean, colors='k')
        ax.axis('equal')
        # ax.set_axis_off()
        ax.xaxis.pane.fill, ax.yaxis.pane.fill, ax.zaxis.pane.fill = False, False, False
        ax.xaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.yaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        ax.zaxis.set_tick_params(labelsize=fontsize*scale, labelfontfamily=fontfamily)
        
        ax.set_xlabel(f'$u$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=10*scale)
        ax.set_ylabel(f'$v$ [m]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
        ax.set_zlabel('$w$ [mm]', fontsize=fontsize*scale, fontfamily=fontfamily, labelpad=5*scale)
        
        # ax.set_xlim([0, 20]);   ax.set_ylim([-2,2]);    ax.set_zlim([-1.5, 1.5]) # 3D
        ax.set_xlim([-1, 1]); ax.set_ylim([-0.5, 0.5]); ax.set_zlim([-0.05, 0.05]) # pca
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_zticklabels([ str(int(float(l.get_text().replace('−', '-'))*1e3)).replace('-', '−') for l in ax.get_zticklabels() ])
        # [ label.set_visible(False) for label in ax.get_xticklabels()[1::2]]
        [ label.set_visible(False) for label in ax.get_yticklabels()[::2]]
        # [ label.set_visible(False) for label in ax.get_zticklabels()[1::2]]
        
        ax.set_box_aspect([2, 1, 0.75])
        
        # sc._offsets3d = (data['X'][idx], data['Y'][idx], data['Z'][idx])
        # sc._A = np.float64(data['Inc'][idx])
        
        # idx = np.logical_and(data_b.index > 0.2, data_b.index <= 0.4)
        # sc._offsets3d = (data_b['X'][idx], data_b['Y'][idx], data_b['Z'][idx])
        # sc._A = data_b['Inc'][idx]
        # # fig, ax = plt.subplots(figsize=(15, 15) )
        
        # ani = animation.FuncAnimation(fig, ani_mov, frames=frames-1, interval=delay*1e3)
        ani = animation.FuncAnimation(fig, ani_mov_wf, frames=frames-1, interval=delay*1e3)

        start = time()
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=int(np.round(1/delay)), metadata=dict(artist='Me'), bitrate=1800)
        ani.save('livox_movement_pcauvw_wire.mp4', writer=writer)
        
        dur = (time() - start) / 60
        print(f'Animation time duration: %.3f min' % (dur) )
        
        
        
        # t0, dt = 0, 0.2
        # idx = np.logical_and( data.index >= t0, data.index < (t0 + dt) )
        
        # cutout = data.iloc[idx]
        # a = np.exp(5 * np.float64(cutout.index))
        # a = a / np.max(a) * 0.8
        
        # ax.scatter(cutout['X'], cutout['Y'], cutout['Z'], s=.1, alpha=a, c=cutout['Inc'])
        # ax.axis('equal')
        # ax.set_axis_off()
        
        
        