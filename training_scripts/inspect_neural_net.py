# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 08:20:23 2020

@author: Jessica Gaines
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import preprocfs as pre
import os
import re

path = 'maeda2task_v_zero/'
tf_seed = 720
np_seed = 1995
dev_proportion = 0.1
un_preprocess_y = [pre.un_scale_0to1]

# retrieve batch size used for training
with open(path + 'results/summary.txt', "r") as f:
    m = re.search('Batch size: (\d+)\n',f.read())
    batch_size = int(m.group(1))

try:
    os.mkdir(path + 'results/' + '/weights')
    os.mkdir(path + 'results/' + '/avg_weights')
    os.mkdir(path + 'results/' + '/loss_dists')
except OSError:
    print ("Error creating directories.")
else:
    print ("Successfully created the directories.")

# import data into dataframe and verify dataframe dimensions
x_train_dev = pd.read_csv(path + 'data/' + 'x_train.csv',index_col=0,header=0)
x_cols = x_train_dev.columns
y_train_dev = pd.read_csv(path + 'data/' + 'y_train.csv',index_col=0,header=0)
y_cols = y_train_dev.columns
x_train_dev_raw = pd.read_csv(path + 'data/' + 'x_train_raw.csv',index_col=0,header=0)
y_train_dev_raw = pd.read_csv(path + 'data/' + 'y_train_raw.csv',index_col=0,header=0)
y_full_raw = pd.read_csv(path + 'data/' + 'y_raw.csv',index_col=0,header=0)
x_test = pd.read_csv(path + 'data/' + 'x_test.csv',index_col=0,header=0)
y_test = pd.read_csv(path + 'data/' + 'y_test.csv',index_col=0,header=0)

# repeatability
tf.random.set_seed(tf_seed)
np.random.seed(np_seed)

# split data
[x_train,y_train,x_dev,y_dev,x_train_raw,y_train_raw,x_dev_raw,y_dev_raw] = pre.split_data(x_train_dev,y_train_dev,data_x_raw=x_train_dev_raw,data_y_raw=y_train_dev_raw,prop=dev_proportion,numpy=True,wrap=False)

# load model
model = tf.keras.models.load_model(path + 'model')
loss = model.evaluate(x_dev,y_dev,batch_size=batch_size,verbose=0)
print('Validation loss: ' + str(np.round(loss,2)))

f = open(path + 'results/' + "summary.txt", "a")
test_loss = model.evaluate(x_test,y_test,batch_size=batch_size,verbose=0)
f.write('\nTest Loss: ' + str(np.round(test_loss,6)))
f.close()
print('Test Loss: ' + str(np.round(test_loss,6)))

# create csv file with inputs (x_inspect), target outputs (y_inspect), and predicted outputs (y_hat)
## pull smaller inspection set out of dev set
insp_proportion = 0.3
insp_max_i = np.round(x_dev.shape[0] * insp_proportion).astype(int)
x_inspect = x_dev[0:insp_max_i]
y_hat = model.predict(x_inspect)
y_inspect = y_dev[0:insp_max_i]
x_raw = x_dev_raw[0:insp_max_i]
y_raw = y_dev_raw[0:insp_max_i]
## convert all to DataFrames   
y_hat_df = pd.DataFrame(y_hat,columns=y_cols)
x_raw_df = pd.DataFrame(x_raw,columns=x_cols)
y_raw_df = pd.DataFrame(y_raw,columns=y_cols)
## inverse preprocess the predicted data
y_hat_raw_df = y_hat_df.copy()
for f in un_preprocess_y:
    y_hat_raw_df = f(y_hat_raw_df,y_full_raw)
## get column labels for predicted and target y columns
y_hat_cols = []
y_insp_cols = []
for col in y_cols:
    y_hat_cols.append(col + ' predicted')
    y_insp_cols.append(col + ' target')
y_hat_raw_df.columns = y_hat_cols
## concatenate dataframes and add loss column
df = pd.concat([x_raw_df,y_hat_raw_df,y_raw_df],axis=1)
df['loss'] = tf.keras.losses.MSE(y_inspect,y_hat)
df.to_csv(path + 'data/' + 'inspection_set.csv')
## plot distribution of losses in inspection set
for i,col in enumerate(df.columns[0:-1]):
    fig = plt.figure()
    plt.scatter(df.iloc[:,i],df['loss'])
    plt.title(col)
    plt.ylabel('mean squared error')
    plt.savefig(path + 'results/' + 'loss_dists/' + col + '_loss_dist.png')
    plt.show()
    
# plot heatmap for average weights
for i,layer in enumerate(model.layers):
    weights = layer.get_weights()[0]
    fig = plt.figure()
    sns.heatmap(weights)
    plt.title('Model weights, layer ' + str(i))
    plt.savefig(path + 'results/' + 'weights/' + 'layer_' + str(i) + '_weights.png')
    plt.show()
    avg_weights = np.loadtxt(path + 'avg_weights/layer_' + str(i) + '.csv', delimiter=",")
    fig = plt.figure()
    sns.heatmap(avg_weights)
    plt.title('Average weights across n trials, layer ' + str(i))
    plt.savefig(path + 'results/' + 'avg_weights/' + 'layer_' + str(i) + '_weights.png')
    plt.show()
    