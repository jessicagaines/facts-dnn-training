# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:43:06 2020

@author: Jessica Gaines
"""

import tensorflow as tf
import pandas as pd
import numpy as np

data_path = 'Model/task2maeda_a_zero_chrontest/'
model_path = 'Model/task2maeda_a_zero/'
step_size = 1

def mse(predicted,actual):
    return np.mean((predicted - actual)**2)

def calc_avg_error(predicted,actual):
    return np.mean(np.abs(predicted - actual))

def scale(point,min,max):
    return (point - min)/(max-min)
def unscale(point,min,max):
    return (point * (max-min) + min)

# read data
scaled_outputs = pd.read_csv(data_path + 'data/y_test.csv',delimiter=',',header=0,index_col=0)
raw_inputs = pd.read_csv(data_path + 'data/x_raw.csv',delimiter=',',header=0,index_col=0)
raw_outputs = pd.read_csv(data_path + 'data/y_raw.csv',delimiter=',',header=0,index_col=0)
walk_start = np.loadtxt(data_path + 'data/walk_start.csv')

# load model
model = tf.keras.models.load_model(model_path + 'model')

# get columns and max/mins for scaling
a_cols = [k for k in raw_inputs.columns if '_dot' not in k]
aa_cols = [k for k in raw_inputs.columns if '_dot' in k and '_dot_dot' not in k]
xxx_cols = [k for k in raw_inputs.columns if '_dot_dot' in k]
a_mins = raw_inputs[a_cols].min()
a_maxs = raw_inputs[a_cols].max()
aa_mins = raw_inputs[aa_cols].min()
aa_maxs = raw_inputs[aa_cols].max()
xxx_mins = raw_inputs[xxx_cols].min()
xxx_maxs = raw_inputs[xxx_cols].max()
aaa_mins = raw_outputs.min()
aaa_maxs = raw_outputs.max()

# calculate next articulator position
## initialize position,velocity,acceleration
mse_losses = []
avg_error = []

predicted_a = pd.DataFrame(index=raw_outputs.index,columns=['jaw','tongue','shape','apex','lip_ht','lip_pr','jaw_dot','tongue_dot','shape_dot','apex_dot','lip_ht_dot','lip_pr_dot'])

for i in raw_outputs.index[0:-1]:
    a = raw_inputs.loc[i,a_cols].to_numpy()
    aa = raw_inputs.loc[i,aa_cols].to_numpy()
    xxx = raw_inputs.loc[i,xxx_cols].to_numpy()
    input_i = np.zeros((len(raw_inputs.columns)))
    input_i[0:len(xxx_cols)] = scale(xxx,xxx_mins,xxx_maxs)
    input_i[len(xxx_cols):len(xxx_cols)+len(a)] = scale(a,a_mins,a_maxs)
    input_i[len(xxx_cols) + len(a):] = scale(aa,aa_mins,aa_maxs)
    input_i = input_i.reshape(1,len(input_i))
    print(input_i)
    predict_aaa = np.squeeze(model.predict(input_i))
    print(predict_aaa)
    scaled_actual_aaa = scaled_outputs.iloc[i,:]
    predict_aaa = unscale(predict_aaa,aaa_mins,aaa_maxs).to_numpy()
    predict_a_next = a + step_size*aa + 0.5*(step_size**2)*predict_aaa
    #predict_a_next = a + step_size*aa
    predict_aa_next = aa + step_size*predict_aaa

    if walk_start[i] > 0:
        predicted_a.loc[i,a_cols] = a
        predicted_a.loc[i,aa_cols] = aa
        mse_losses.append(0)
        avg_error.append(0)
    if walk_start[i+1] == 0:
        predicted_a.loc[i+1,a_cols] = predict_a_next
        predicted_a.loc[i+1,aa_cols] = predict_aa_next
        a_next = raw_inputs.loc[i+1,a_cols].to_numpy()
        aa_next = raw_inputs.loc[i+1,aa_cols].to_numpy()
        predicted = predicted_a.loc[i+1,:]
        actual = np.concatenate([a_next,aa_next])
        loss = mse(predicted,actual)
        error = calc_avg_error(predicted,actual)
        mse_losses.append(loss)
        avg_error.append(error)
predicted_a['walk_start'] = walk_start
predicted_a['mse_loss'] = mse_losses
predicted_a['avg_error'] = avg_error
predicted_a.to_csv(data_path + 'results/predicted_positions.csv')
print('Mean squared error: ')
print(np.mean(mse_losses))

