import numpy as np
import matplotlib.pyplot as plt
import pywt
import pyarrow.parquet as pq
import pandas as pd

import scipy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import matthews_corrcoef, make_scorer, roc_auc_score
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter

from tqdm import tqdm
from numba import jit

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=scipy.stats.entropy(probabilities)
    return entropy
 
@jit('float32(float32[:])')
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]
 
def get_features(list_values):
    entropy = calculate_entropy(list_values)
    statistics = calculate_statistics(list_values)
    return [entropy] + statistics

def get_uci_har_features(dataset, labels, waveletname):
    uci_har_features = []
    for signal_no in tqdm(range(0, len(dataset))):
        features = []
        signal = dataset[signal_no, :]
        list_coeff = pywt.wavedec(signal, waveletname)
        for coeff in list_coeff:
            features += get_features(coeff)
        uci_har_features.append(features)
    X = np.array(uci_har_features)
    Y = np.array(labels)
    return X, Y


#X_index, Y_data = np.array([], dtype=np.int), np.array([])
#meta_set = pd.read_csv('data/metadata_train.csv')
#signal_id = meta_set.loc[(meta_set.target == 1), 'signal_id']
#X_index = np.append(X_index, signal_id.values)
#Y_data = np.ones(len(X_index))
#Y_data = np.append(Y_data, np.zeros(len(X_index)) )
#
#signal_id = meta_set.loc[(meta_set.target == 0), 'signal_id']
#X_index = np.append(X_index, signal_id.values[:len(X_index)])
#
#train_set = pq.read_table('data/train.parquet', columns=[str(i) for i in X_index]).to_pandas()
#signal = np.array(train_set).T.reshape(-1, 800000)

X_data, Y_data = np.loadtxt('X_data.txt'), np.loadtxt('Y_data.txt') #get_uci_har_features(signal, Y_data, 'db4')
X_data, Y_data = shuffle(X_data, Y_data)

#np.savetxt('X_data.txt', X_data)
#np.savetxt('Y_data.txt', Y_data)

#X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.3, shuffle=True)
#
#min_max_scaler = MinMaxScaler()
##y_test, y_train = np.array(y_test, dtype='int'), np.array(y_train, dtype='int')
#
#ma_depth:60, b_estimators:1000
#clf = RandomForestRegressor(n_estimators=1000, max_depth=60)

mc = make_scorer(matthews_corrcoef)
#cv5 = cross_val_score(clf, X_data, Y_data, scoring=mc, cv=5)
#cv6 = cross_val_score(clf, X_data, Y_data, scoring=mc, cv=6)
#cv7 = cross_val_score(clf, X_data, Y_data, scoring=mc, cv=7)
#
#cv_mean = [np.mean(cv5), np.mean(cv6), np.mean(cv7)]
#cv_std = [np.std(cv5), np.std(cv6), np.std(cv7)]
#
#for i in range(len(cv_mean)):
#    print("cv",i+5,": ", cv_mean[i], " +/- ", cv_std[i])
#print("cv: ", np.mean(cv_mean), " +/- ", np.mean(cv_std))

ra = make_scorer(roc_auc_score)
#3000, max_depth: 50
#clf = ExtraTreesRegressor()

clf = MLPClassifier()
param_grid = { 
      'solver': ['lbfgs'],
      'max_iter': [2000, 3000], 
      'alpha': 10.0 ** -np.arange(1, 4), 
      'hidden_layer_sizes':np.arange(2000, 3000, 500), 
      'activation': ['tanh',  'relu']
}

CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring=mc)
print("start fitiing...")
CV_rfc.fit(X_data, Y_data)
bp = CV_rfc.best_params_
bs = CV_rfc.best_score_
print(bp)
print(bs)