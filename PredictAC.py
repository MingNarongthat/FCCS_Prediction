import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
from fcsfiles import ConfoCor3Raw   # First, <pip install fcsfiles>
import multipletau as mtau

# And the tf and keras framework, thanks to Google
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

ts = 200    # time-step in nanoseconds for making bins for correlation analysis
# Multi-tau params
dt = 2e-7   # Bin width for correlation
m_val = 8    # Log register size
# Load files
# Read raw fcs file (photon arrival times)
os.getcwd()
path_data1 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Testing', '*_ChS1.raw')
path_data2 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Testing', '*_ChS2.raw')
dirlist = glob.glob(path_data1) + glob.glob(path_data2)
num_batches = 5


model_type = 'DNN'
model_name = 'train_Dense_newopt_test22222'


# model_name2 = 'train_LSTM_test'
rawfile = dirlist[6]


raw_data = ConfoCor3Raw(rawfile)
times = raw_data.asarray()
# Binning
if model_type == 'DNN':
    bin_num = 5000000
elif model_type == 'LSTM':
    bin_num = 10000
else:
    sys.exit('No model trained')

timesb, bin_counts = raw_data.asarray(bins=bin_num)
# pad zeros till binnum size is attained
bin_counts = np.insert(bin_counts, bin_counts.size, np.zeros(bin_num - bin_counts.size), axis=0)

# Load model
model = load_model('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/{}'.format(model_name))
# model2 = load_model('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/{}'.format(model_name2))

# Test using input data
X_AC = bin_counts.astype(float)
Y_AC = mtau.autocorrelate(X_AC, m=m_val, deltat=dt, normalize=True)

if model_type == 'DNN':
    X_test = np.reshape(bin_counts, (1, -1))
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

# X_test2 = model2.predict(X_test, batch_size=num_batches)
# ynn2 = Y_test[1:len(Y_test)][1]
# ynn2 = ynn2[1:len(ynn2)]
yac = Y_AC[1:len(Y_AC[:, 0]), 1]
xac = Y_AC[1:len(Y_AC[:, 0]), 0]

plt.plot(xac, yac, label='GT')
# plt.plot(xac, ynn2, 'g*:', label='Model Prediction normal')
plt.plot(xac, ynn, 'r--', label='Model Prediction')
plt.legend()
plt.title("{}".format(model_name), fontweight='bold')
plt.xscale('log')
plt.xlabel('lag time (sec)', fontweight='bold')
plt.ylabel('ACF', fontweight='bold')
# plt.show()
plt.savefig(model_name+'.png', dpi=1200)
print("Neural Network AC and multi-tau AC correlation (test): ", np.corrcoef(ynn, yac)[1][0])

# clear session
del(ynn)
del(yac)
del(Y_pred)
