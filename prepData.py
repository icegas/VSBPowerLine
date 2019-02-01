import pyarrow.parquet as pq
import pandas as pd
import pywt

import numpy as np
import sys
from tqdm import tqdm

import scipy.stats as sts
from scipy import signal
from ex import waveletSmooth

df_train = pd.read_csv('data/metadata_train.csv')

df_train = df_train.set_index(['id_measurement', 'phase'])
df_train.head()

max_num = 127
min_num = -128

sample_size = 800000

def min_max_transf(ts, min_data, max_data, range_needed=(-1,1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data - min_data)
    if range_needed[0] < 0:    
        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def transform_ts(ts, n_dim=200, min_max=(-1,1)):
    # convert data into -1 to 1
    ts_std = min_max_transf(ts, min_data=min_num, max_data=max_num)
    # bucket or chunk size, 5000 in this case (800000 / 160)
    bucket_size = int(sample_size / n_dim)
    # new_ts will be the container of the new data
    new_ts = []
    # this for iteract any chunk/bucket until reach the whole sample_size (800000)
    for i in range(0, sample_size, bucket_size):
        # cut each bucket to ts_range
        ts_range = ts_std[i:i + bucket_size]
        ts_range = waveletSmooth(ts_range, level=1)
        # calculate each feature
        mean = ts_range.mean()
        std = ts_range.std() # standard deviation
        std_top = mean + std # I have to test it more, but is is like a band
        std_bot = mean - std
        # I think that the percentiles are very important, it is like a distribuiton analysis from eath chunk
        percentil_calc = np.percentile(ts_range, [0, 1, 25, 50, 75, 99, 100]) 
        max_range = percentil_calc[-1] - percentil_calc[0] # this is the amplitude of the chunk
        relative_percentile = percentil_calc - mean # maybe it could heap to understand the asymmetry

        k4 = sts.kstat(ts_range, 4)
        kurts = sts.kurtosis(ts_range)
        skew = sts.skew(ts_range)
        mode = sts.mode(ts_range)
        iqr = sts.iqr(ts_range)
        vartion = sts.variation(ts_range)
        gm = sts.gmean(ts_range)
        peaks, _ = (signal.find_peaks(ts_range, distance=20))
        num_peaks = len(peaks)
        width_of_peaks = signal.peak_widths(ts_range, peaks, rel_height=0.3)
        width_of_peaks = width_of_peaks[0]
        mean_w_p = width_of_peaks.mean()
        min_w_p = np.min(width_of_peaks)
        max_w_p = np.max(width_of_peaks)
        peak_prom = signal.peak_prominences(ts_range, peaks)
        mean_prom = peak_prom[0].mean()

        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range,
        k4, kurts, skew, mode, iqr, vartion, gm, num_peaks, width_of_peaks, mean_w_p, min_w_p, max_w_p, mean_prom]),
        percentil_calc, relative_percentile]))
    return np.asarray(new_ts)

def prep_data(start, end):
    # load a piece of data from file
    praq_train = pq.read_pandas('data/train.parquet', columns=[str(i) for i in range(start, end)]).to_pandas()
    X = []
    y = []
    # using tdqm to evaluate processing time
    # takes each index from df_train and iteract it from start to end
    # it is divided by 3 because for each id_measurement there are 3 id_signal, and the start/end parameters are id_signal
    for id_measurement in tqdm(df_train.index.levels[0].unique()[int(start/3):int(end/3)]):
        X_signal = []
        # for each phase of the signal
        for phase in [0,1,2]:
            # extract from df_train both signal_id and target to compose the new data sets
            signal_id, target = df_train.loc[id_measurement].loc[phase]
            # but just append the target one time, to not triplicate it
            if phase == 0:
                y.append(target)
            # extract and transform data into sets of features
            X_signal.append(transform_ts(praq_train[str(signal_id)]))
        # concatenate all the 3 phases in one matrix
        X_signal = np.concatenate(X_signal, axis=1)
        # add the data to X
        X.append(X_signal)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y

def load_all():
    total_size = len(df_train)
    first_sig = df_train.index[0][0]

    n_parts = 20
    part_size = int(total_size / n_parts)
    last_part = total_size % n_parts

    start_end = [[x, x+part_size] for x in range(first_sig, total_size + first_sig, part_size)]
    start_end = start_end[:-1] + [[start_end[-1][0], start_end[-1][0] + last_part]]
    print(start_end)
    i = 19

    for ini, end in start_end[19:]: #[(0, int(total_size/4)), (int(total_size/4), int(total_size/2)), ( int(total_size/2) , int(3*total_size/4) ), 
                    #( int(3*total_size/4), int(total_size) )]:

        X_temp, y_temp = prep_data(ini, end)

        #X = np.concatenate(X)
        #y = np.concatenate(y)
        np.save("Y_data/y_denoized"+str(i)+".npy",y_temp)
        np.save("X_data/X_denoized"+str(i)+".npy",X_temp)
        i = i + 1
        del X_temp, y_temp
        #X.append(X_temp)
        #y.append(y_temp)

load_all()

