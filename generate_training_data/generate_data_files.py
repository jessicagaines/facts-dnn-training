'''
@author: Jessica Gaines
'''
import numpy as np
import time
import maeda as mda
from makeTrainingData import find_artic_params
import os
import sys

#if len(sys.argv) < 2:
#    print("Usage: python generate_data_files.py <number of values per input>")
#    sys.exit(0)
#try:
#    n = int(sys.argv[1])
#except:
#    print("Usage: python generate_data_files.py <number of values per input>")
#    sys.exit(0)

start = time.time()

path = 'training_data_files_TB75-180'
if not os.path.isdir(path):
    try:
        os.mkdir(path)
    except OSError:
        print('Creation of directory %s failed.' %path)
        path = os.getcwd()
        
n=5
TC = np.array([1,1,0,0], 'float32')
PC = np.array([0.00114,35000,1600,1.5,300000], 'float32')
anc = 0.0
palateCon=np.loadtxt("palate_contour.txt")

row_dump = 10000
print_update = 100
values = np.linspace(-3,3,n)
i = 0
j = 0
AM_array = np.zeros((row_dump,7))
formant_array = np.zeros((row_dump,5))
vocal_tract_array = np.zeros((row_dump,6))
print_flag = 0
for jaw in values:
    for tongue in values:
        for shape in values:
            for apex in values:
                for lip_ht in values:
                    for lip_pr in values:
                        for larynx in values:
                            AM = np.array([jaw,tongue,shape,apex,lip_ht,lip_pr,larynx], 'float32')
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
                            if i > 0 and i % row_dump == 0:
                                np.savetxt(path + '/AM_' + str(j) + '.csv',AM_array,delimiter=',')
                                np.savetxt(path + '/formants_' + str(j) + '.csv',formant_array,delimiter=',')
                                np.savetxt(path + '/vocal_tract_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
                                print(str(i) + ' rows saved.')
                                j += 1
                                
i = i % row_dump
AM_array = AM_array[0:i]
formant_array = formant_array[0:i]
vocal_tract_array = vocal_tract_array[0:i]
np.savetxt(path + '/AM_' + str(j) + '.csv',AM_array,delimiter=',')
np.savetxt(path + '/formants_' + str(j) + '.csv',formant_array,delimiter=',')
np.savetxt(path + '/vocal_tract_' + str(j) + '.csv',vocal_tract_array,delimiter=',')
print(str(i) + ' rows saved. Time: ' + str(time.time()-start))