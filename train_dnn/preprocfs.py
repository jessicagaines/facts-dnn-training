# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 10:14:51 2020

@author: Jessica Gaines
"""
import numpy as np
from varname.helpers import Wrapper
import pandas as pd

def read_in_data(read_path):
    AMs = pd.read_csv(read_path + 'AM_0.csv',header=None,index_col=None)
    AMs.columns = ['jaw','tongue','shape','apex','lip_ht','lip_pr','larynx'][0:AMs.shape[1]]
    formants = pd.read_csv(read_path + 'formants_0.csv',header=None,names=range(5))
    task_params = pd.read_csv(read_path + 'vocal_tract_0.csv',header=None,names=['TT_Den','TT_Alv','TB_Pal','TB_Vel','TB_Pha','LPRO','LA'])
    try: walk_start = np.loadtxt(read_path + 'walk_start_marker_0.csv',delimiter=',',)
    except: walk_start = [] 
    i = 1
    while True:
        try:
            new_walk_start = np.loadtxt(read_path + 'walk_start_marker_' + str(i) + '.csv',delimiter=',',)
            walk_start = np.append(walk_start,new_walk_start)
        except: pass
        try:
            new_AMs = pd.read_csv(read_path + 'AM_' + str(i) + '.csv',header=None,names=AMs.columns)
            new_formants = pd.read_csv(read_path + 'formants_' + str(i) + '.csv',header=None,names=formants.columns)
            new_task_params = pd.read_csv(read_path + 'vocal_tract_' + str(i) + '.csv',header=None,names=task_params.columns)
        except:
            break
        AMs = AMs.append(new_AMs)
        formants = formants.append(new_formants)
        task_params = task_params.append(new_task_params)
        i += 1 
    AMs.reset_index(drop=True, inplace=True)
    formants.reset_index(drop=True, inplace=True)
    formants.columns = ['F1','F2','F3','F4','F5']
    formants = formants.iloc[:,0:3]
    AMs = AMs.iloc[:,0:6]
    task_params.reset_index(drop=True, inplace=True)
    try: walk_start = np.loadtxt(read_path + 'walk_start_marker.csv',delimiter=',',)
    except: pass
    return AMs,formants,task_params,walk_start

def normalize(df):
    norm = (df/np.mean(df)) * np.mean(np.mean(df))
    return norm
def diff(df):
    df.columns=range(0,df.shape[1])
    df_diff = df[[0]].copy()
    for i in range(0,4):
        df_diff[i+1] = df[i+1] - df[i]
    df_diff.columns = ['F1','F2-F1','F3-F2','F4-F3','F5-F4']
    return df_diff
def normalize_inc_diff(df):
    norm = normalize(df)
    df_diff = diff(df)
    for i in range(1,df_diff.shape[1],1):
        col = str(norm.columns[i]) + '_diff'
        norm[col] = df_diff[i]
    return norm
def inverse(df):
    inverse = (1/df) * np.mean(np.mean(df))
    return inverse

def scale_0to1(df):
    mins = df.min()
    maxs = df.max()
    scaled = ((df-mins) / (maxs-mins))
    return scaled

def un_scale_0to1(df,df_raw):
    mins = df_raw.min()
    maxs = df_raw.max()
    unscaled = (df * (maxs-mins)) + mins
    return unscaled

def tscore(df):
    means = df.mean()
    stds = df.std()
    stds_sqrtn = stds/np.sqrt(df.shape[0])
    tscore = (df - means) / stds_sqrtn
    return tscore
def conv_to_percent(df):
    sums = df.sum(axis=1)
    percents_t = df.T/sums
    percents = percents_t.T
    return percents

def add_nth_deriv(df, walk_start, how_at_start=None,deriv=1,timestep=1):
    df_dot = df.copy()
    df_dot.reset_index(drop=True,inplace=True)
    cols = df.columns
    if deriv > 1:
        cols = [k for k in cols if '_dot' in k]
    for col in cols:
        col_dot = col + '_dot'
        if how_at_start == 'gradient':
            df_dot.loc[:,col_dot] = np.gradient(df[col],timestep)
        else:
            if how_at_start == 'zero':
                df_dot.loc[0,col_dot] = 0
            elif how_at_start == 'rm':
                df_dot.loc[0,col_dot] = np.nan
            for i in df_dot.index[1:]:
                if walk_start[i] > 0 and how_at_start == 'zero':
                    df_dot.loc[i,col_dot] = 0
                elif walk_start[i] > 0 and how_at_start == 'rm':
                    df_dot.loc[i,col_dot] = np.nan
                else:
                    val = df_dot.loc[i,col]
                    prev_val = df_dot.loc[i-1,col]
                    df_dot.loc[i,col_dot] = (val - prev_val)/timestep
        df_dot.dropna(axis=0,how='any',inplace=True)
    return df_dot

def split_data(data_x,data_y,prop,data_x_raw=pd.DataFrame(),data_y_raw=pd.DataFrame(),wrap=False,numpy=False):
    shuffled_index = np.arange(data_x.shape[0])
    np.random.shuffle(shuffled_index)
    len_test_set = np.round(data_x.shape[0] * prop).astype(int)
    test_index = shuffled_index[0:len_test_set]
    training_index = shuffled_index[len_test_set:]
    ## assign data to each set
    x_test = Wrapper(data_x.iloc[test_index])
    y_test = Wrapper(data_y.iloc[test_index])
    x_train = Wrapper(data_x.iloc[training_index])
    y_train = Wrapper(data_y.iloc[training_index])
    if not data_x_raw.empty and not data_y_raw.empty:
        ## also split raw inputs
        x_train_raw = Wrapper(data_x_raw.iloc[training_index])
        x_test_raw = Wrapper(data_x_raw.iloc[test_index])
        y_train_raw = Wrapper(data_y_raw.iloc[training_index])
        y_test_raw = Wrapper(data_y_raw.iloc[test_index])
        sets = [x_train,y_train,x_test,y_test,x_train_raw,y_train_raw,x_test_raw,y_test_raw]
    else:
        sets = [x_train,y_train,x_test,y_test]
    if numpy:
        formatted = []
        for set in sets:
            set.value = set.value.to_numpy()
            formatted.append(set)
    if wrap:
        return sets
    else:
        unwrapped = []
        for set in sets:
            unwrapped.append(set.value)
        return unwrapped