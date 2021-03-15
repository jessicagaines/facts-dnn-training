# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 14:09:16 2020

@author: Jessica Gaines
"""
import pandas as pd
import numpy as np
import preprocfs as pre
import os
import shutil

inputs = 'task_params'
outputs = 'AMs'
preprocess_x = [pre.scale_0to1]
preprocess_y = [pre.scale_0to1]
add_derivs = {'x' : ['gradient','gradient'], 
              'y' : ['gradient','gradient']}
jacobian = True
test_proportion = 10/100
np_seed = 1995
read_path = '../training_data_100000_restricted/'
write_path = 'task2maeda_a_gradient_100000_restricted/'
timestep = 0.005

try:
    shutil.rmtree(write_path)
except:
    print(write_path + ' does not exist.')
else:
    print('Remove existing folder ' + write_path)

try:
    os.mkdir(write_path)
    os.mkdir(write_path + 'data/')
    os.mkdir(write_path + 'model/')
    os.mkdir(write_path + 'results/')
    os.mkdir(write_path + 'avg_weights/')
except OSError:
    print ("Error creating " + write_path)
else:
    print ("Successfully created the directory %s " % write_path)

# import data into dataframe and verify dataframe dimensions
AMs,formants,task_params,walk_start = pre.read_in_data(read_path)

# assign inputs and outputs
if inputs == 'AMs': x_pre = AMs
# if inputs == 'AMs': x_pre = AMs.join(formants)
if inputs == 'formants': x_pre = formants
if inputs == 'task_params': x_pre = task_params
if outputs == 'AMs' : y_pre = AMs
if outputs == 'formants' : y_pre = formants
if outputs == 'task_params' : y_pre = task_params

for i,how in enumerate(add_derivs.get('x')):
    x_pre = pre.add_nth_deriv(x_pre, walk_start, how, deriv=i+1,timestep=timestep)
for i,how in enumerate(add_derivs.get('y')):
    y_pre = pre.add_nth_deriv(y_pre, walk_start, how, deriv=i+1,timestep=timestep)
    
if jacobian:
    cols = y_pre.columns
    cols = [k for k in cols if '_dot_dot' in k]
    y_new = y_pre[cols].copy()
    
    for i,marker in enumerate(walk_start):
        if marker > 0:
            y_pre.iloc[i-1,:] = np.nan
            x_pre.iloc[i,:] = np.nan
            y_new.iloc[i,:] = np.nan
            walk_start[i-1] = np.nan
    y_pre.dropna(axis=0,how='any',inplace=True)
    x_pre.dropna(axis=0,how='any',inplace=True)
    y_new.dropna(axis=0,how='any',inplace=True)
    walk_start = walk_start[~np.isnan(walk_start)]
    
    cols = x_pre.columns
    cols = [k for k in cols if '_dot_dot' in k]
    x_dot_dot = x_pre[cols]
    x_dot_dot.reset_index(drop=True,inplace=True)
    cols = y_pre.columns
    cols = [k for k in cols if '_dot_dot' not in k]
    a_and_a_dot = y_pre[cols]
    a_and_a_dot.reset_index(drop=True,inplace=True)
    x_new = pd.concat([x_dot_dot,a_and_a_dot],axis=1)
    
    x_pre = x_new
    y_pre = y_new
    x_pre.reset_index(drop=True,inplace=True)
    y_pre.reset_index(drop=True,inplace=True)
    

x_pre.to_csv(write_path + 'data/' + 'x_raw.csv')
y_pre.to_csv(write_path + 'data/' + 'y_raw.csv')
# preprocessing
x = x_pre.copy()
for f in preprocess_x:
    x = f(x)
y = y_pre.copy()
print(y)
for f in preprocess_y:
    y = f(y)
print(y)

# repeatability
np.random.seed(np_seed)
# split data into testing, development, and training sets
if test_proportion > 0 and test_proportion < 1:
    ## determine list of indexes that will be assigned to each set
    sets = pre.split_data(x,y,test_proportion,x_pre,y_pre,wrap=True)
    # check data split correctly
    print('Inputs complete set: ' + str(x.shape))
    print('Outputs complete set: ' + str(y.shape))
    # save split sets to file
    for set in sets:
        set.value.reset_index(drop=True,inplace=True)
        print(set.name + ' ' + str(set.value.shape))
        set.value.to_csv(write_path + 'data/' + set.name + '.csv')
elif test_proportion == 0:
    x.to_csv(write_path + 'data/x_train.csv')
    y.to_csv(write_path + 'data/y_train.csv')
    walk_start.savetxt(write_path + 'data/walk_start.csv')
elif test_proportion == 1:
    x.to_csv(write_path + 'data/x_test.csv')
    y.to_csv(write_path + 'data/y_test.csv')   
    np.savetxt(write_path + 'data/walk_start.csv', walk_start)
