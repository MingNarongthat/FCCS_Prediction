# ================================================================
# FCCS Project
# Sysmex Coporate, CRL, Leading Medical Research Group
# Written by Narongthat (Ming) Thanyawet, 2022 September
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from fcsfiles import ConfoCor3Raw
import multipletau as mtau
from Pre_process import pre_data

# And the tf and keras framework, thanks to Google
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

# Multi-tau params
dt = 2e-7   # Bin width for correlation
m_val = 8    # Log register size

# Read raw fcs file (photon arrival times)
os.getcwd()
path_data1 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Testing', '*_ChS1.raw')
path_data2 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Testing', '*_ChS2.raw')
dirlist = glob.glob(path_data1) + glob.glob(path_data2)

# setting parameters -------------------------------------------------------------------------
num_batches = 5     # for DNN: almost default 10, but 'Both' case shold be 2. LSTM shold be 5

in_data = 'single'  # single or many (features)
model_type = 'DNN'      # DNN or LSTM
model_name = 'train_DNN_3ly'     # model name
plot_type = 'Show'          # Show or Save plot
cut_num = 0     # cutting number out 0 (no cut), 1 (cut 0), 2 (cut 0 1), 3 (cut 0 1 2)

# rawfile = dirlist[9]        # testing data
# print(rawfile)
# --------------------------------------------------------------------------------------------
for rawfile in dirlist:
    raw_data = ConfoCor3Raw(rawfile)
    times = raw_data.asarray()

    # Binning
    if model_type == 'DNN':
        bin_num = 5000000
    elif model_type == 'LSTM':
        bin_num = 10000
    else:
        sys.exit('No model trained')

    bin_counts_cut, bin_counts = pre_data(rawfile, cut_num, 'BinCut')
    # np.savetxt('testparam.csv', bin_counts_cut, delimiter=',')

    # Load model
    model = load_model('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/{}'.format(model_name))

    # Test using input data
    X_AC = bin_counts.astype(float)
    Y_AC = mtau.autocorrelate(X_AC, m=m_val, deltat=dt, normalize=True)

    if model_type == 'DNN':
        if in_data == 'single':
            X_test = np.reshape(bin_counts_cut, (1, -1))
        else:
            X_test = bin_counts_cut

        Y_pred = model.predict(X_test).flatten()
        ynn = Y_pred[1:len(Y_pred)]

    elif model_type == 'LSTM':
        # input for LSTM model for prediction
        X_test = np.zeros([num_batches, bin_num])
        for i in range(num_batches):
            X_test[i] = bin_counts
        Y_pred = model.predict(X_test, batch_size=num_batches)
        ynn = Y_pred[1:len(Y_pred)][1]
        ynn = ynn[1:len(ynn)]

    else:
        sys.exit('No model trained')

    yac = Y_AC[1:len(Y_AC[:, 0]), 1]
    xac = Y_AC[1:len(Y_AC[:, 0]), 0]

    plt.plot(xac, yac, label='GT')
    plt.plot(xac, ynn, 'r--', label='Model Prediction')

    plt.legend()
    plt.title("{}".format(model_name), fontweight='bold')

    plt.xscale('log')
    plt.xlabel('lag time (sec)', fontweight='bold')
    plt.ylabel('ACF', fontweight='bold')
    if plot_type == 'Show':
        pass
        # plt.show()
    elif plot_type == 'Save':
        plt.savefig(model_name + '.png', dpi=1200)
    print("Neural Network AC and multi-tau AC correlation (test): ", np.corrcoef(ynn, yac)[1][0])

    # clear session
    del(ynn)
    del(yac)
    del(Y_pred)
