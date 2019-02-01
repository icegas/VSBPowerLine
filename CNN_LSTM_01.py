import keras
import keras.backend as K
from keras.layers import LSTM,Dropout,Dense,TimeDistributed,Conv1D,MaxPooling1D,Flatten
from keras.models import Sequential
import tensorflow as tf

import pyarrow.parquet as pq
import pandas as pd
import pywt

import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingRegressor
from sklearn.metrics import confusion_matrix

from numba import jit
from ex import maddest, waveletSmooth

from tqdm import tqdm

@jit('float32(float32[:,:], int32)')
def feature_extractor(x, n_part=1000):
    lenght = len(x)
    pool = np.int32(np.ceil(lenght/n_part))
    output = np.zeros((n_part,))
    for j, i in enumerate(range(0,lenght, pool)):
        if i+pool < lenght:
            k = x[i:i+pool]
        else:
            k = x[i:]
        output[j] = np.max(k, axis=0) - np.min(k, axis=0)
    return output

def keras_auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''

    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

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
#
#cv_mean = np.array([])
#cv_std = np.array([])
#eph_mean = np.array([])
#
#features = [600]
##levels = [1]
##wavelets = ['db32', 'db4']  #pywt.wavelist(kind='discrete')[:3]
##for wavelet in wavelets:
##    for k in levels:
#for feature in features:
#    X_data = np.array([])
#    for i in tqdm(range(signal.shape[0])):
#
#        db4 = waveletSmooth(signal[i], 1, wavelet='db4')
#        X_data = np.append(X_data, np.abs(feature_extractor(db4, n_part=feature)))
#
#    #features = features*len(wavelets)*len(levels)
#
#    X_data = np.reshape(X_data, (-1, feature))
#    #X_data = X_data.reshape(-1,X_data[0].shape[0])
#
#    kf = KFold(n_splits=5, shuffle=True)
#    kf.get_n_splits(X_data)
#    cv = np.array([])
#    cv_tmp = np.array([])
#    eph_idx = np.array([])

    #for train_index, test_index in kf.split(X_data):
    #    X_train, X_test = X_data[test_index], X_data[train_index]
    #    y_train, y_test = Y_data[test_index], Y_data[train_index]
    #X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.8, random_state=42)
    ##y_train, y_test = np.array(y_train, dtype=int), np.array(y_test, dtype=int)
    #clf = ExtraTreesClassifier(300)
    #clf.fit(X_train, y_train)
    #pred = matthews_corrcoef(y_test, (clf.predict(X_test) > 0.7) *1 )
    #print(pred)
    #    n_signals = 1 #So far each instance is one signal. We will diversify them in next step
    #    n_outputs = 1 #Binary Classification

    #    #Build the model
    #    verbose, epochs, batch_size = True, 60, 64
    #    n_steps, n_length = int(feature/10), 10
    #    X_train = X_train.reshape((X_train.shape[0], n_steps, n_length, n_signals))
    #    X_test = X_test.reshape((X_test.shape[0], n_steps, n_length, n_signals))
    #    y_train = np.reshape(y_train, (-1,) )
    #    # define model
    #    model = Sequential()
    #    model.add(TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_signals)))
    #    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

    #    model.add(TimeDistributed(Conv1D(filters=256, kernel_size=3, activation='relu')))
    #    model.add(TimeDistributed(Dropout(0.5)))
    #    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))

    #    model.add(TimeDistributed(Flatten()))
    #    model.add(LSTM(200))
    #    model.add(Dropout(0.5))
    #    model.add(Dense(512, activation='relu'))
    #    model.add(Dense(n_outputs, activation='sigmoid'))


    #    model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=[matthews_correlation])

    #    for i in range(epochs):
    #        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=verbose)

    #        pred = model.predict(X_test)
    #        pred = np.reshape(pred, -1)
    #        pred = (pred > 0.7) * 1

    #        acc = accuracy_score(y_test, pred)
    #        math_corr = matthews_corrcoef(y_test, pred)
    #        print(i)
    #        print ("accuracy: ", acc)
    #        print ("matthews corr: ", math_corr)
    #        cv_tmp = np.append(cv, math_corr)
    #    eph_idx = np.append(eph_idx, np.argmax(cv_tmp))
    #    cv = np.append(cv, np.argmax(cv_tmp))

    #cv_mean = np.append(cv_mean, cv.mean())
    #cv_std = np.append(cv_std, cv.std())
    #eph_mean = np.append(eph_mean, eph_idx.mean())
    #print ("mean:", cv.mean(), " +/- ", cv.std())

#cv_mean = np.reshape(cv_mean, (len(wavelets), len(levels)))
#cv_std = np.reshape(cv_std, (len(wavelets), len(levels)))

#for i in range(len(features)):
#    print (features[i], ": ", eph_mean[i], " mean:", cv_mean[i], " +/- ", cv_std[i])


#plt.show()