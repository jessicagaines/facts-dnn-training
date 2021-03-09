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

    #TT40-70
    #TB75-180
    TTmin = 0
    TTmax = 301
    TBmin = 350
    TBmax = 1400
    TTind = np.where(area == min(area[TTmin:TTmax]))[0][0]
    TBind = np.where(area == min(area[TBmin:TBmax]))[0][0]
    TTCL=(TTind+400)/10
    TTCD=min(area[TTmin:TTmax])*mm2cm
    TBCL=(TBind+400)/10
    TBCD=min(area[TBmin:TBmax])*mm2cm    
    
    ##LA
    LA = (external_y[28] -internal_y[28])*mm2cm
    #print('LA = ' + str(LA))

    ##LP
    LP = (external_x[27] -external_x[28])*mm2cm
    #print('LP = ' + str(LP))
    
    if verbose:
        print('TTCL = ' + str(TTCL))
        print('TTCD = ' + str(TTCD))
        print('TBCL = ' + str(TBCL))
        print('TBCD = ' + str(TBCD))
        print('LA = ' + str(LA))
        print('LP = ' + str(LP))

    #GLO for pitch?
    TTx_values = [pCon_x[TTind*10], newTx[TTind]]
    TTy_values = [pCon_y[TTind*10], newTy[TTind]]
    TBx_values = [pCon_x[TBind*10], newTx[TBind]]
    TBy_values = [pCon_y[TBind*10], newTy[TBind]]
    #V = np.array([[newTx[0] ,newTy[0]],[newTx[4000],newTy[4000]]])
    #PV= np.array([[pCon_x[0] ,pCon_y[0]],[pCon_x[2000],pCon_y[2000]],[pCon_x[4000],pCon_y[4000]]])
    #V = np.array([[newTx[0] ,newTy[0]],[newTx[2000],newTy[2000]],[newTx[4000],newTy[4000]]])

    #uncomment these lines to check the alignment of vectors
    #PV= np.array([[pCon_x[0] ,pCon_y[0]],[pCon_x[4000],pCon_y[4000]],[pCon_x[8000],pCon_y[8000]],[pCon_x[12000],pCon_y[12000]],[pCon_x[14000],pCon_y[14000]]])
    #V = np.array([[newTx[0] ,newTy[0]],[newTx[4000],newTy[4000]],[newTx[8000],newTy[8000]],[newTx[12000],newTy[12000]],[newTx[14000],newTy[14000]]])
    #origin = [0], [0] # origin point

    if plot:
        plt.plot(internal_x,internal_y,'o')
        # Graph:
        plt.plot(interpolated_points[0:n_points,0],interpolated_points[0:n_points,1])
        #plt.plot(inter, 'ok', label='original points');
        plt.axis('equal'); plt.xlabel('x'); plt.ylabel('y');
        # Plot data with 1.0 max limit in y.
        plt.figure()
        # Set x axis limit.

        # plot area func
        #plt.plot(area)
        #plt.show()
        # Plot points.

        #plt.plot(pCon_x,pCon_y,Tx,Ty,newTx,newTy)
        plt.plot(external_x-ref[0],external_y-ref[1],pCon_x,pCon_y,Tx,Ty,newTx,newTy)
        plt.plot(TTx_values, TTy_values,'r')
        plt.plot(TBx_values, TBy_values,'r')
        plt.axis('equal')
        plt.show()

        #plt.hist(err, bins = 100)
        #plt.quiver(*origin, PV[:,0], PV[:,1], angles='xy', scale_units='xy', scale=1)
        #plt.quiver(*origin, V[:,0], V[:,1], angles='xy', scale_units='xy', scale=1)
        #plt.plot(pCon_x,pCon_y,Tx,Ty,newTx,newTy)
        #plt.show()
    return [TTCL,TTCD,TBCL,TBCD,LA,LP]
    
find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=True,verbose=True)