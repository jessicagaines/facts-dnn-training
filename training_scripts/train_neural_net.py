# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 10:24:56 2020

@author: Jessica Gaines
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import preprocfs as pre

# define file paths
path = 'task2maeda_a_gradient/'
data_path = path + 'data/'
model_path = path + 'model' # path to save or load new model to/from

# Hyperparameters and model architecture
epochs = 1000
batch_size = 128
hidden_layer_sizes = [128,128,128,128,128]
hidden_layer_activation = ['relu', 'relu','relu','relu','relu']
learning_rate = 0.0001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss_function = 'mse'
reg_penalty = 0
dev_proportion = 10/100
min_delta = 0.001 # Early Stopping criteria
tf_seed = 720
np_seed = 1995
n = 10 # number of trials in validation
early_stop = False

# import data into dataframe and verify dataframe dimensions
x_train_dev = pd.read_csv(data_path + 'x_train.csv',index_col=0,header=0)
x_cols = x_train_dev.columns
y_train_dev = pd.read_csv(data_path + 'y_train.csv',index_col=0,header=0)
y_cols = y_train_dev.columns

# repeatability
tf.random.set_seed(tf_seed)
np.random.seed(np_seed)

# split data
[x_train,y_train,x_dev,y_dev] = pre.split_data(x_train_dev,y_train_dev,dev_proportion,numpy=True)

# repeatability
tf.random.set_seed(tf_seed)
np.random.seed(np_seed)

# build model
model = tf.keras.Sequential()
model.add(tf.keras.Input(x_train.shape[1],))
for size,act in zip(hidden_layer_sizes,hidden_layer_activation):
    model.add(tf.keras.layers.Dense(size,activation=act,kernel_regularizer=tf.keras.regularizers.l1(reg_penalty)))
model.add(tf.keras.layers.Dense(y_train.shape[1]))
model.compile(optimizer=optimizer, loss=loss_function)

# train model
start = time.time()
if early_stop:
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, min_delta=min_delta,patience=100)
    history = model.fit(x_train,y_train,validation_data=(x_dev,y_dev),batch_size=batch_size,epochs=epochs,verbose=False,callbacks=[es])
else:
    history = model.fit(x_train,y_train,validation_data=(x_dev,y_dev),batch_size=batch_size,epochs=epochs,verbose=False)
end = time.time()
model.save(model_path)

# plot training metrics
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss over course of training')
plt.xlabel('epoch')
plt.ylabel('loss')
min_loss = np.min(history.history['val_loss'])
max_loss = np.max(history.history['val_loss'])
print(max_loss)
plt.ylim([0,1.1*max_loss])
plt.legend()
plt.savefig(path + 'results/training_loss.png')
plt.show()
hrs = np.floor((end-start) / 3600)
mins = np.floor(((end-start) % 3600 / 60))
secs = np.round((end-start) % 60)
f = open(path + 'results/' + "summary.txt", "w")
f.write("Architecture\n")
f.write(str(hidden_layer_sizes) + '\n')
f.write(str(hidden_layer_activation) + '\n')
f.write('Learning Rate: ' + str(learning_rate)+ '\n')
f.write('Batch size: ' + str(batch_size) + '\n')
f.write('Data: ' + data_path + '\n')
f.write('Training Time: ' + str(int(hrs)) + 'hr ' + str(int(mins)) + 'min ' + str(int(secs)) + 'sec' + '\n')
f.write('Initial Validation Loss During Training: ' + str(np.round(history.history['val_loss'][0],6)) + '\n')
f.write('Min Validation Loss During Training: ' + str(np.round(np.min(history.history['val_loss']),6)) +  '\n')
f.write('\n')
f.close()
print("Architecture")
print(hidden_layer_sizes)
print(hidden_layer_activation)
print('Learning Rate: ' + str(learning_rate))
print('Batch size: ' + str(batch_size))
print('Data: ' + data_path)
print('Training Time: ' + str(int(hrs)) + 'hr ' + str(int(mins)) + 'min ' + str(int(secs)) + 'sec')
print('Min Validation Loss During Training: ' + str(np.round(np.min(history.history['val_loss']),6)))
    
#evaluate model
## prep data structures
losses = []
marginal_losses = np.zeros((n,y_dev.shape[1]))
weights_acc = list()
for layer in model.layers:
    weights = layer.get_weights()[0]
    weights_acc.append(np.zeros(weights.shape))
## train model n times
for i in range(0,n,1):
    print(i)
    # re-split data into training and dev sets
    [x_train,y_train,x_dev,y_dev] = pre.split_data(x_train_dev,y_train_dev,dev_proportion,numpy=True)
    # train model
    model.fit(x_train,y_train,
              validation_data=(x_dev,y_dev),
              batch_size=batch_size,
              epochs=epochs,
              verbose=False)
              #callbacks=[es])
    loss = model.evaluate(x_dev,y_dev,batch_size=batch_size,verbose=0)
    losses.append(loss)

    # evaluate individual formants
    y_hat = model.predict(x_dev)
    for j in range(0,y_dev.shape[1],1):
        # target and predicted for one formant at a time
        y_true = y_dev[:,j]
        y_pred = y_hat[:,j]
        # (target - predicted)^2, average over all values in the dev set
        marginal_loss = tf.keras.losses.MSE(y_true,y_pred) # returns scalar
        marginal_losses[i,j] = marginal_loss
    # save weights
    for i,layer in enumerate(model.layers):
        weights = layer.get_weights()[0]
        weights_acc[i] = weights_acc[i] + weights

if n>0:
    # compute average weights
    for i,w in enumerate(weights_acc):
        avg_weights = w/n
        np.savetxt(path + 'avg_weights/layer_' + str(i) + '.csv', avg_weights, delimiter=",")
    # print summary of validation
    print('Validation loss: ' + str(np.round(np.mean(losses),6)) + ' +/- ' + str(np.round(np.std(losses),6)) + ' across ' + str(n) + ' trials.')
    error_means = np.mean(marginal_losses,axis=0)
    error_stds = np.std(marginal_losses,axis=0)
    target_means = np.mean(y_dev,axis=0)
    f = open(path + 'results/' + "summary.txt", "a")
    f.write('Validation loss: ' + str(np.round(np.mean(losses),6)) + ' +/- ' + str(np.round(np.std(losses),6)) + ' across ' + str(n) + ' trials.' + '\n')    
    f.write('Marginal losses:' + '\n')
    print('Marginal losses')
    for j in range(0,y_dev.shape[1],1):
        string = y_cols[j] + ' mean error: ' + str(np.round(error_means[j],6)) + ' +/- ' + str(np.round(error_stds[j],6))
        f.write(string + '\n')
        print(string)
        #print('\tNormalized by mean: ' + str(np.round(error_means[j]/target_means[j],6)) + ' +/- ' + str(np.round(error_stds[j]/target_means[j],6)))
    f.close()

        