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
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model, save_model

# ------------------------------------- setting parameters and paths for dataset
os.getcwd()
path_data1 = os.path.join('/home/guest/ming_internship/Training', '*_ChS1.raw')
path_data2 = os.path.join('/home/guest/ming_internship/Training', '*_ChS2.raw')
dirlist = glob.glob(path_data1) + glob.glob(path_data2)

start_time = datetime.datetime.now()
model_name = 'train_DNN_adam_layer6_128'

# FCS Experiment conditions
meas_time = 10      # measurement time in seconds
ts = 200  # time-step in nanoseconds for making bins for correlation analysis

# Multi-tau params
dt = 2e-7       # Bin width for correlation
m_val = 8        # Log register size

# Train params
epoch_num = 10      # number of epochs
num_batches = 10
file_counter = 0
dirlist1 = dirlist[0:5]
for files in dirlist1:
    # ------------------------------------- read data from binary file (the signal)
    raw_data = ConfoCor3Raw(files)     # info of data of diffusion time
    times = raw_data.asarray()      # diffusion time data
    bin_num = 5000000       # setting binned number
    time_sb, bin_counts = raw_data.asarray(bins=bin_num)        # time of data and binned signal count

    # insert zero until the number of data equal bin_num = 5,000,000
    bin_counts = np.insert(bin_counts, bin_counts.size, np.zeros(bin_num-bin_counts.size), axis=0)
    bin_counts = [i for i in bin_counts if i > 2]
    # ------------------------------------- preparing input data for model by shuffle the data
    data_in = np.zeros([num_batches, bin_num])
    plt.plot(bin_counts)
plt.show()
# bin_counts = [i for i in bin_counts if i != 0]
# diff = len(times) - len(bin_counts)
# times = times[:len(times)-diff]
# print(len(bin_counts))
# print(len(times))
# plt.scatter(times, bin_counts)
# plt.show()
