'''
@author: Jessica Gaines
'''
import numpy as np
import time
import maeda as mda
import pandas as pd
from makeTrainingData import find_artic_params
import os

start = time.time()

read_path = 'training_data_files_TB75-180'
write_path = 'training_data_files_TB75-180_explore_valid'
if not os.path.isdir(write_path):
    try:
        os.mkdir(write_path)
    except OSError:
        print('Creation of directory %s failed.' %write_path)

# read in set of valid points        
AMs = pd.read_csv(read_path + '/AM_0.csv', header=None,names=['jaw','tongue','shape','apex','lip_ht','lip_pr','larynx'])
i = 1
while True:
    try:
        AM = pd.read_csv(read_path + '/AM_' + str(i) + '.csv',header=None,names=['jaw','tongue','shape','apex','lip_ht','lip_pr','larynx'])
    except:
        break
    AMs = AMs.append(AM)
    i += 1
AMs.reset_index(drop=True,inplace=True)
# set constants
n = 1
step = 0.4
TC = np.array([1,1,0,0], 'float32')
PC = np.array([0.00114,35000,1600,1.5,300000], 'float32')
anc = 0.0
palateCon=np.loadtxt("palate_contour.txt")
row_dump = 5000
print_update = 100
# initialize data structures
AM_array = np.zeros((row_dump,7))
formant_array = np.zeros((row_dump,5))
vocal_tract_array = np.zeros((row_dump,6))
print_flag = 0
# iterate through each valid point
i = 0
prev_i = 0
j = 0
np.random.seed(100)
rows = np.array(AMs.index.copy())
np.random.shuffle(rows)
count = 0
for row in rows:
    print(count)
    count += 1
    params = AMs.loc[row,:]
    # iterate through each parameter
    for param in range(len(params)):
        # how many steps up/down?
        for k in range(1,n+1):
            up = params.copy()
            up[param] = up[param] + k*step
            down = params.copy()
            down[param] = down[param] - k*step
            # for each new set of parameters, run through model to find formants
            for AM in [up,down]:
                formant,internal_x,internal_y,external_x,external_y= mda.maedaplant(5,29,29,29,29,TC,PC,AM,anc)
                if min(formant)>100 and max(formant)<10000:
                    vocal_tract = find_artic_params(internal_x,internal_y,external_x,external_y,palateCon,plot=False,verbose=False)
                    AM_array[i%row_dump] = AM
                    formant_array[i%row_dump] = formant
                    vocal_tract_array[i%row_dump] = vocal_tract
                    i += 1
                    end = time.time()
                if i % print_update == 0 and print_flag == 1:
                    print(str(i) + ' valid points processed. Time: ' + str(time.time()-start))
                    start = time.time()
                    print_flag = 0
                if print_flag == 0 and i % print_update != 0:
                    print_flag = 1
                if i > prev_i and i % row_dump == 0:
                    np.savetxt(write_path + '/AM_' + str(j) + '.csv',AM_array,delimiter=',')
                    np.savetxt(write_path + '/formants_' + str(j) + '.csv',formant_array,delimiter=',')
                    np.savetxt(write_path + '/vocal_tract_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
                    print(str(i) + ' rows saved.')
                    prev_i = i
                    j += 1
i = i % row_dump
AM_array = AM_array[0:i]
formant_array = formant_array[0:i]
vocal_tract_array = vocal_tract_array[0:i]
np.savetxt(write_path + '/AM_' + str(j) + '.csv',AM_array,delimiter=',')
np.savetxt(write_path + '/formants_' + str(j) + '.csv',formant_array,delimiter=',')
np.savetxt(write_path + '/vocal_tract_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
print(str(i) + ' rows saved. Time: ' + str(time.time()-start))