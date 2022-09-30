# ================================================================
# FCCS Project
# Sysmex Coporate, CRL, Leading Medical Research Group
# Written by Narongthat (Ming) Thanyawet, 2022 September
# ================================================================

import numpy as np
import os
import glob
from fcsfiles import *  # First, <pip install fcsfiles>

os.getcwd()
path_data1 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Training', '*_ChS1.raw')
path_data2 = os.path.join('/Users/mingnarongthat/Documents/Ph.D./Sysmex/ming_internship/Training', '*_ChS2.raw')
dirlist = glob.glob(path_data1) + glob.glob(path_data2)


def pre_data(dirlist, cut_set, type_data):
    if cut_set == 1:
        num_data = 100000
    elif cut_set == 2:
        num_data = 2800
    elif cut_set == 3:
        num_data = 200
    elif cut_set == 0:
        num_data = 5000000
    else:
        print("Wrong scenario!!!")

    # if cut_set == 1 & type_data == 'Both':
    #     num_data = 60000

    # ------------------------------------- read data from binary file (the signal)
    raw_data = ConfoCor3Raw(dirlist)     # info of data of diffusion time
    times = raw_data.asarray()      # diffusion time data
    bin_num = 5000000       # setting binned number
    time_sb, bin_counts = raw_data.asarray(bins=bin_num)        # time of data and binned signal count

    # insert zero until the number of data equal bin_num = 5,000,000
    bin_counts = np.insert(bin_counts, bin_counts.size, np.zeros(bin_num-bin_counts.size), axis=0)

    # ------------------------------------- preparing input data for model by shuffle the data
    # data_in = np.zeros([num_batches, bin_num])
    bin_counts_cut = [i for i in bin_counts if i >= cut_set]
    bin_counts_cut = np.array(bin_counts_cut)

    if len(bin_counts_cut) <= num_data:
        bin_counts_cut = np.insert(bin_counts_cut, bin_counts_cut.size, np.zeros(num_data - bin_counts_cut.size), axis=0)
    else:
        bin_counts_cut = bin_counts_cut[:num_data]

    times = np.array(times, dtype=np.float32)
    if len(times) <= 60000:
        test1 = np.insert(times, times.size, np.zeros(60000 - times.size), axis=0)
    else:
        test1 = times[:60000]

    if type_data == 'Both':
        both_data = np.empty([2, len(test1)])

        for i in range(2):
            if i == 0:
                both_data[i][:] = bin_counts_cut
            elif i == 1:
                both_data[i][:] = test1
            else:
                break

    if type_data == 'BinCut':
        return bin_counts_cut, bin_counts
    elif type_data == 'Times':
        return test1, bin_counts
    elif type_data == 'Both':
        return both_data, bin_counts


if __name__ == '__main__':
    cut_set = 0
    type_data = 'BinCut'
    bin_cut, bin_counts = pre_data(dirlist, cut_set, type_data)
