# ================================================================
# FCCS Project
# Sysmex Coporate, CRL, Leading Medical Research Group
# Written by Narongthat (Ming) Thanyawet, 2022 September
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import datetime
from fcsfiles import *  # First, <pip install fcsfiles>
import multipletau
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout, Activation
from keras.models import Sequential, load_model, save_model
from Pre_process import pre_data


# ----------------------------------- DNN architecture aspiration from Transformer model
# Inspiration from Transformer model in paper 2017 that change from RNN to transformer
# Attention Is All You Need: https://arxiv.org/abs/1706.03762
def dnn_encode_decode():
    model = Sequential()
    model.add(Dense(units=num_AC, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(units=85, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Dense(units=num_AC, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc', 'mae'])
    model.summary()
    return model


# ------------------------------------- LSTM and Dense layer from paper IEEE2018
# Deep Recurrent Neural Networks for Prostate Cancer Detection: Analysis of Temporal Enhanced Ultrasound
# https://ieeexplore.ieee.org/abstract/document/8395313
def lstm_dense():
    model = Sequential()
    model.add(LSTM(units=5, batch_input_shape=(X_train.shape[0], X_train.shape[1], 1), return_sequences=True))
    model.add(LSTM(units=5, batch_input_shape=(X_train.shape[0], X_train.shape[1], 1), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_AC))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mae'])
    model.summary()
    return model


# --------------------------------------- Multi-layer perceptron architecture
# paper: Time Series Classification from scratch with deep nearal networks: A strong baseline, 2016
def mlp():
    model = Sequential()
    model.add(Dense(units=num_AC, input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.1))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=500, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=num_AC, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Activation('softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc', 'mae'])
    model.summary()
    return model

# ------------------------------------- setting parameters and paths for dataset
os.getcwd()
path_data1 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Training', '*_ChS1.raw')
path_data2 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Training', '*_ChS2.raw')
dirlist = glob.glob(path_data1) + glob.glob(path_data2)

start_time = datetime.datetime.now()
model_name = 'train_DNN_3ly'

# Multi-tau params
dt = 2e-7       # Bin width for correlation
m_val = 8        # Log register size

# Train params
epoch_num = 10      # number of epochs
num_batches = 5     # for DNN: almost default 10, but 'Both' case shold be 2. LSTM shold be 5
file_counter = 0
cut_num = 0     # cutting number out 0 (no cut), 1 (cut 0), 2 (cut 0 1), 3 (cut 0 1 2)
type_case = 'BinCut'  # BinCut, Times, Both
model_type = 'DNN'  # DNN, LSTM,

# dirlist = dirlist[0:1]
for files in dirlist:
    print('Processing file ', file_counter+1, '/', len(dirlist), ' ', files)
    # ------------------------------------- read data from binary file (the signal)
    bin_counts_cut, bin_counts = pre_data(files, cut_num, type_case)

    if type_case == 'Both':
        pass
    else:
        # ------------------------------------- preparing input data for model by shuffle the data
        # create zeros array for input data for model [num_batches * bin_num]
        data_in = np.zeros([num_batches, len(bin_counts_cut)])

        # ------------------------------------- increase data by shifting/inverse the data
        for i in range(num_batches):
            if i == 0:
                data_in[i] = bin_counts_cut
                data_in[i+1] = bin_counts_cut[::-1]
                i = i+2
                continue
            else:
                shifted_frame = np.insert(bin_counts_cut, 0, np.zeros(i*100), axis=0)
                data_in[i] = shifted_frame[0:len(bin_counts_cut)]
                shifted_frame_rev = shifted_frame[::-1]
                data_in[i+1] = shifted_frame_rev[0:len(bin_counts_cut)]
                i = i+2
                if i > (num_batches-1):
                    break
                continue

    # -------------------------------------- calculate auto-correlation function of signal
    corr_f1 = multipletau.autocorrelate(np.float64(bin_counts), m=m_val, deltat=dt, normalize=True)
    num_AC = len(corr_f1[:, 0])

    if type_case == 'Both':
        Y_train = corr_f1.reshape(2, 85)
    else:
        # -------------------------------------- prepare X_train and Y_train
        Y_train = np.zeros([num_batches, num_AC])

        for i in range(num_batches):
            corr_fn = multipletau.autocorrelate(np.float64(bin_counts), m=m_val, deltat=dt, normalize=True)
            Y_train[i] = corr_fn[0:len(corr_fn[:, 0]), 1]

    if type_case == 'Both':
        X_train = bin_counts_cut
    else:
        X_train = data_in

    keras.backend.clear_session()

    # --------------------------------------- fir the model for training
    if file_counter == 0:
        model = dnn_encode_decode()
    else:
        model = load_model(model_name)

    if model_type == 'DNN':
        history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=16, validation_split=0.3, callbacks=[])
    else:
        history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=num_batches)

    model.save(model_name)
    file_counter = file_counter + 1

# ---------------------------------------- calculate the time used
end_time = datetime.datetime.now()
total_use = (end_time - start_time).total_seconds()
print("Training done. Time elapsed: ", total_use, "s")
