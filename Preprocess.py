import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from fcsfiles import *  # First, <pip install fcsfiles>

# ------------------------------------- setting parameters and paths for dataset
os.getcwd()
path_data1 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Training', '*_ChS1.raw')
path_data2 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Training', '*_ChS2.raw')
dirlist = glob.glob(path_data1) + glob.glob(path_data2)

# Train params
num_batches = 10
file_counter = 0
dirlist1 = dirlist[0:5]
cut_set = 2

if cut_set == 1:
    num_data = 100000
elif cut_set == 2:
    num_data = 2800
elif cut_set == 3:
    num_data = 200
else:
    print("Wrong scenario!!!")


for files in dirlist:
    # ------------------------------------- read data from binary file (the signal)
    raw_data = ConfoCor3Raw(files)     # info of data of diffusion time
    times = raw_data.asarray()      # diffusion time data
    bin_num = 5000000       # setting binned number
    time_sb, bin_counts = raw_data.asarray(bins=bin_num)        # time of data and binned signal count

    # insert zero until the number of data equal bin_num = 5,000,000
    bin_counts = np.insert(bin_counts, bin_counts.size, np.zeros(bin_num-bin_counts.size), axis=0)

    # ------------------------------------- preparing input data for model by shuffle the data
    data_in = np.zeros([num_batches, bin_num])
    bin_counts_cut = [i for i in bin_counts if i >= cut_set]
    bin_counts_cut = np.array(bin_counts_cut)

    if len(bin_counts_cut) <= num_data:
        bin_counts_cut = np.insert(bin_counts_cut, bin_counts_cut.size, np.zeros(num_data - bin_counts_cut.size), axis=0)
    else:
        bin_counts_cut = bin_counts_cut[:num_data]

    # diff = len(times) - len(bin_counts)
    # times = times[:len(times) - diff]

    # print(len(bin_counts))
    # print(len(bin_counts_cut))
    # plt.scatter(times, bin_counts)
    # plt.show()
    plt.plot(bin_counts_cut)

    # file_counter = file_counter + len(bin_counts_cut)

plt.show()
# print('This is average bin count: {}'.format(round(file_counter/70)))
