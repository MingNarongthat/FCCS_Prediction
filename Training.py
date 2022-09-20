import numpy as np
import pandas as pd
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


# ----------------------------------- DNN architecture aspiration from Transformer model
# Inspiration from Transformer model in paper 2017 that change from RNN to transformer
# Attention Is All You Need: https://arxiv.org/abs/1706.03762
def dnn_encode_decode():
    model = Sequential()
    model.add(Dense(units=num_AC, input_shape=(X_train.shape[1],), activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.3))
    # model.add(Dense(units=32, activation='sigmoid'))
    # model.add(Dropout(0.3))
    # model.add(Dense(units=16, activation='sigmoid'))
    # model.add(Dropout(0.3))
    # model.add(Dense(units=32, activation='sigmoid'))
    # model.add(Dropout(0.1))
    # model.add(Dense(units=64, activation='sigmoid'))
    # model.add(Dropout(0.2))
    model.add(Dense(units=num_AC, activation='relu'))
    # opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.4)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc', 'mae'])
    model.summary()
    return model


# ------------------------------------- LSTM and Dense layer from paper IEEE2018
# Deep Recurrent Neural Networks for Prostate Cancer Detection: Analysis of Temporal Enhanced Ultrasound
# https://ieeexplore.ieee.org/abstract/document/8395313
def lstm_dense():
    model = Sequential()
    model.add(LSTM(units=5, batch_input_shape=(X_train.shape[0], X_train.shape[1], 1), return_sequences=True))
    # model.add(LSTM(units=64, return_sequences=True, batch_input_shape=(X_train.shape[0], X_train.shape[1], 1)))
    # model.add(LSTM(units=32, return_sequences=True, batch_input_shape=(X_train.shape[0], X_train.shape[1], 1)))
    model.add(LSTM(units=5, batch_input_shape=(X_train.shape[0], X_train.shape[1], 1), stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(units=num_AC))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mae'])
    model.summary()
    return model


# --------------------------------------- pure LSTM layers, but the architecture used encoder-decoder
def lstm():
    model = Sequential()
    model.add(LSTM(units=num_AC, return_sequences=True, dropout=0.3, recurrent_dropout=0.1, input_shape=(X_train.shape[1],), stateful=False))
    model.add(LSTM(units=64, return_sequences=True, dropout=0.3, recurrent_dropout=0.1, stateful=False))
    model.add(LSTM(units=32, return_sequences=True, dropout=0.3, recurrent_dropout=0.1, stateful=False))
    model.add(LSTM(units=64, return_sequences=True, dropout=0.3, recurrent_dropout=0.1, stateful=False))
    model.add(LSTM(units=num_AC, return_sequences=True, dropout=0.3, recurrent_dropout=0.1, stateful=False))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
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
model_name = 'train_Dense_newopt_test111111'

# FCS Experiment conditions
# meas_time = 10      # measurement time in seconds
ts = 200  # time-step in nanoseconds for making bins for correlation analysis

# Multi-tau params
dt = 2e-7       # Bin width for correlation
m_val = 8        # Log register size

# Train params
epoch_num = 10      # number of epochs
num_batches = 10
file_counter = 0
save_loss = []
# dirlist = dirlist[0:2]
for files in dirlist:
    print('Processing file ', file_counter+1, '/', len(dirlist), ' ', files)
    # ------------------------------------- read data from binary file (the signal)
    raw_data = ConfoCor3Raw(files)     # info of data of diffusion time
    times = raw_data.asarray()      # diffusion time data
    bin_num = 5000000       # setting binned number
    time_sb, bin_counts = raw_data.asarray(bins=bin_num)        # time of data and binned signal count

    # insert zero until the number of data equal bin_num = 5,000,000
    bin_counts = np.insert(bin_counts, bin_counts.size, np.zeros(bin_num-bin_counts.size), axis=0)

    # ------------------------------------- preparing input data for model by shuffle the data
    data_in = np.zeros([num_batches, bin_num])   # create zeros array for input data for model [num_batches * bin_num]

    # ------------------------------------- increase data by shifting/inverse the data
    for i in range(num_batches):
        if i == 0:
            data_in[i] = bin_counts
            data_in[i+1] = bin_counts[::-1]
            i = i+2
            continue
        else:
            shifted_frame = np.insert(bin_counts, 0, np.zeros(i*100), axis=0)
            data_in[i] = shifted_frame[0:bin_num]
            shifted_frame_rev = shifted_frame[::-1]
            data_in[i+1] = shifted_frame_rev[0:bin_num]
            i = i+2
            if i > (num_batches-1):
                break
            continue
    # plt.plot(data_in)
    # plt.show()

    # -------------------------------------- calculate auto-correlation function of signal
    corr_f1 = multipletau.autocorrelate(data_in[0], m=m_val, deltat=dt, normalize=True)
    # xac = corr_f1[1:len(corr_f1[:, 0]), 0]    # time
    # yac = corr_f1[1:len(corr_f1[:, 0]), 1]    # correlation function

    # plt.plot(xac, yac)
    # plt.xscale('log')
    # plt.show()
    num_AC = len(corr_f1[:, 0])

    # -------------------------------------- prepare X_train and Y_train
    Y_train = np.zeros([num_batches, num_AC])

    for i in range(num_batches):
        corr_fn = multipletau.autocorrelate(data_in[i], m=m_val, deltat=dt, normalize=True)
        Y_train[i] = corr_fn[0:len(corr_fn[:, 0]), 1]

    X_train = data_in
    # print(Y_train.shape)
    keras.backend.clear_session()

    # --------------------------------------- fir the model for training
    if file_counter == 0:
        model = dnn_encode_decode()
    else:
        model = load_model(model_name)
    # history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=16, validation_split=0.3, callbacks=[])
    history = model.fit(X_train, Y_train, epochs=epoch_num, batch_size=num_batches)
    model.save(model_name)
    if file_counter == 0:
        save_loss = history.history['loss']
    else:
        save_loss.append(history.history['loss'])
    file_counter = file_counter + 1

    if file_counter == len(dirlist):
        # plt.plot(history.epoch, np.array(history.history['val_loss']), 'r--', label='Val loss')
        plt.plot(history.epoch, np.array(history.history['loss']), label='Train loss')
        plt.savefig('Losses.png')
        plt.close()

# ---------------------------------------- calculate the time used
np.savetxt('SGD_loss.csv', save_loss, delimiter=',')
end_time = datetime.datetime.now()
total_use = (end_time - start_time).total_seconds()
print("Training done. Time elapsed: ", total_use, "s")



