# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 02:46:27 2020

@author: Jessica Gaines
"""

import numpy as np
import time
import maeda as mda
from makeTrainingData import find_artic_params
import os

full_start = time.time()

write_path = 'training_data_3articsmove'
if not os.path.isdir(write_path):
    try:
        os.mkdir(write_path)
    except OSError:
        print('Creation of directory %s failed.' %write_path)

def is_valid_formant(formant):
    #if min(formant) > 100 and max(formant)<10000:
    if formant[0]>250 and formant[0]<900 and min(formant) > 0 and max(formant)<10000:
        return True
    else:
        return False

def is_valid_config(AM):
    if max(AM) > 3:
        return False
    if min(AM) < -3:
        return False
    return True

# set constants
max_step = 0.25
max_n_steps = 50
min_n_points = 100000
AM_dims = 6
task_dims = 7
TC = np.array([1,1,0,0], 'float32')
PC = np.array([0.00114,35000,1600,1.5,300000], 'float32')
anc = 0.0
palateCon=np.loadtxt("palate_contour.txt")
row_dump = 5000
print_update = 10
# initialize data structures
AM_array = np.zeros((row_dump,AM_dims))
formant_array = np.zeros((row_dump,5))
vocal_tract_array = np.zeros((row_dump,task_dims))
walk_start_marker = np.zeros(min_n_points)
i = 0
prev_i = 0
j = 0
walks = 0
# repeatability
np.random.seed(150)
while i < min_n_points:
    # find starting point
    start = time.time()
    AM = np.zeros(7,dtype="float32") 
    rand = np.random.random(size=AM_dims)
    scaled_rand = (rand * 6) - 3
    AM[0:AM_dims] = scaled_rand
    formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
    count = 0
    walk_start_marker[i] = 1
    # randomly walk
    while is_valid_formant(formant) and count <= max_n_steps:
        # store valid params
        count += 1
        vocal_tract = find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=False,verbose=False)
        AM_array[i%row_dump] = AM[0:AM_dims]
        formant_array[i%row_dump] = formant
        vocal_tract_array[i%row_dump] = vocal_tract
        i += 1
        if i > prev_i and i % row_dump == 0:
            np.savetxt(write_path + '/AM_' + str(j) + '.csv',AM_array,delimiter=',')
            np.savetxt(write_path + '/formants_' + str(j) + '.csv',formant_array,delimiter=',')
            np.savetxt(write_path + '/vocal_tract_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
            print(str(i) + ' rows saved.')
            prev_i = i
            j += 1
        # take a step
        rand = np.random.random(size=3)
        pos = np.random.choice(range(6),3)
        scaled_rand = (rand * max_step*2) - max_step
        AM[pos] += scaled_rand
        formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
        
    end = time.time()
    if count > 0:
        walks += 1
        print('Walk ' + str(walks) + ' completed with ' + str(count) + ' points.  ' + str(round(end-start,2)) + ' s runtime.')

walk_start_marker_full = np.zeros(i)
walk_start_marker_full[0:min_n_points] = walk_start_marker
np.savetxt(write_path + '/walk_start_marker.csv',walk_start_marker_full,delimiter=',')

i = i % row_dump
AM_array = AM_array[0:i]
formant_array = formant_array[0:i]
vocal_tract_array = vocal_tract_array[0:i]
np.savetxt(write_path + '/AM_' + str(j) + '.csv',AM_array,delimiter=',')
np.savetxt(write_path + '/formants_' + str(j) + '.csv',formant_array,delimiter=',')
np.savetxt(write_path + '/vocal_tract_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
print(str(i) + ' rows saved. Total time: ' + str(time.time()-full_start))