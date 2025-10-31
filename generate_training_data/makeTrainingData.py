# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 22:08:53 2020
@author: Kwang Seob Kim
@edited by: Jessica Gaines
"""

import numpy as np
import maeda as mda
import matplotlib.pyplot as plt
import time
from scipy.interpolate import interp1d
import relokate as rlk

TC = np.array([1,1,0,0], 'float32')
PC = np.array([0.00114,35000,1600,1.5,300000], 'float32')
#AM = np.array([1.51,0.47,2.13,1.65,-0.09,2.10,0], 'double') 
#AM = np.array([-1.6127,-2.2717,-0.4682,-0.0173,0,0,0], 'double')
AM = np.array([0,0,0,0,0,0,0], 'float32') 
#AM = np.array([0.5, -2.0, 1.0, -2.0,  1.0, -1.0, 0.0], 'float32') #iy
#AM = np.array([0.0, -1.0, 1.0, -2.0,  1.0, -1.0, 0.0], 'float32') #ey
#AM = np.array([-1.0,  0.0, 1.0, -2.0,  1.0, -0.5, 0.0], 'float32') #eh
#AM = np.array([ -1.5,  0.5, 0.0, -0.5,  0.5, -0.5, 0.0], 'float32') #ah

#AM = np.array([-1.5,  2.0, 0.0, -0.5,  0.5, -0.5, 0.0], 'float32') #aa
#AM = np.array([-0.4,  3.0, 1.5,  0.0, -0.3,  0.0, 0.0], 'float32') #ao
#AM = np.array([-.7,  3.0, 1.5,  0.0, -0.6,  0.0, 0.0], 'float32') #oh
#AM = np.array([0.5,  2.0, 1.5, -2.0, -1.0,  1.5, 0.0], 'float32') #uw
#AM = np.array([0.5, -1.0, 1.0, -2.0, -0.5,  1.0, 0.0], 'float32') #iw
#AM = np.array([0.0, -0.2, 1.0, -1.5, -0.25, 0.5, 0.0], 'float32') #ew
#AM = np.array([-1.0, -0.5, 0.5, -2.0,  0.2, -0.5, 0.0], 'float32') #oe

#AM = np.array([-3.0, -3.0, -3.0, -3.0, 1.5, -3.0, -3.0], 'float32') #error

anc = 0.0

formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
#print(formant)
palateCon=np.loadtxt("palate_contour.txt")

def find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=True,verbose=True):
    pCon_x = palateCon[0,]
    pCon_y = palateCon[1,]
    
    mm2cm = 10
    ## need to import angle data here too
    inter = np.array([internal_x,internal_y]).T
    if inter[-1].all() == inter[-2].all():
        inter = inter[0:-1]
    # Linear length along the line:
    distance = np.cumsum( np.sqrt(np.sum( np.diff(inter, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    
    n_points = 50000
    alpha = np.linspace(0, 1, n_points)
    interpolator =  interp1d(distance, inter, kind='cubic', axis=0)
    interpolated_points = interpolator(alpha)
    
    ref= np.array([10,10])
    Tx = interpolated_points[0:n_points,0]-ref[0]
    Ty = interpolated_points[0:n_points,1]-ref[1]

    rlk_start = time.time()
    
    newTx, newTy, err = rlk.res2center(1401,1401,1401,pCon_x,pCon_y,Tx,Ty)
    rlk_end = time.time()
    if verbose:
        print(rlk_end - rlk_start)
        print(np.mean(err))
        print(np.max(err))
        
    area=np.zeros(1400,'double')
    for x in range(0, 1400):
        area[x] = np.linalg.norm([pCon_x[x*10]-newTx[x], pCon_y[x*10]-newTy[x]])  
    
    CD_Den = np.linalg.norm([pCon_x[290]-newTx[29], pCon_y[290]-newTy[29]])*mm2cm #42.9 deg
    CD_Alv = np.linalg.norm([pCon_x[1800]-newTx[180], pCon_y[1800]-newTy[180]])*mm2cm #58.0
    CD_Pal = np.linalg.norm([pCon_x[5240]-newTx[524], pCon_y[5240]-newTy[524]])*mm2cm #92.41
    CD_Vel = np.linalg.norm([pCon_x[8110]-newTx[811], pCon_y[8110]-newTy[811]])*mm2cm #121.1
    CD_Pha = np.linalg.norm([pCon_x[13980]-newTx[1398], pCon_y[13980]-newTy[1398]])*mm2cm #179.82

    ##LA
    LA = (external_y[28] -internal_y[28])*mm2cm

    ##LP
    LP = (external_x[27] -external_x[28])*mm2cm
    
    if verbose:
        print('TT_Den = ' + str(CD_Den))
        print('TT_Alv = ' + str(CD_Alv))
        print('TB_Pal = ' + str(CD_Pal))
        print('TB_Vel = ' + str(CD_Vel))
        print('TB_Pha = ' + str(CD_Pha))
        print('LA = ' + str(LA))
        print('LP = ' + str(LP))
        
    if plot:
        plt.plot(internal_x,internal_y,'o')
        # Graph:
        plt.plot(interpolated_points[0:n_points,0],interpolated_points[0:n_points,1])
        #plt.plot(inter, 'ok', label='original points');
        plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y');
        # Plot data with 1.0 max limit in y.
        plt.figure()
        # Set x axis limit.

    return [CD_Den,CD_Alv,CD_Pal,CD_Vel,CD_Pha,LA,LP]
    
#find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=True,verbose=True)