import pywt
import numpy as np
import pyarrow.parquet as pq
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from features import avg_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential


def plot_wavelet(time, signal, scales, 
                 waveletname = 'morl', 
                 cmap = plt.cm.seismic, 
                 title = 'Wavelet Transform (Power Spectrum) of signal', 
                 ylabel = 'Period (years)', 
                 xlabel = 'Time'):
    
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    power = (abs(coefficients)) ** 2
    period = 1. / frequencies
    levels = [0.0625, 0.125, 0.25, 0.5, 0.75, 1, 1.25, 1.5]
    contourlevels = np.log2(levels)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    im = ax.contourf(time, np.log2(period), np.log2(power), contourlevels, extend='both',cmap=cmap)
    
    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)
    
    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], -1)
    
    cbar_ax = fig.add_axes([0.95, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()

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



X_index, Y_data = np.array([], dtype=np.int), np.array([])
meta_set = pd.read_csv('data/metadata_train.csv')
signal_id = meta_set.loc[(meta_set.target == 1), 'signal_id']
X_index = np.append(X_index, signal_id.values[:100])
Y_data = np.ones(len(X_index))
Y_data = np.append(Y_data, np.zeros(len(X_index)) )

signal_id = meta_set.loc[(meta_set.target == 0), 'signal_id']
X_index = np.append(X_index, signal_id.values[:100])#len(X_index)])

train_set = pq.read_table('data/train.parquet', columns=[str(i) for i in X_index]).to_pandas()
signal = np.array(train_set).T.reshape(-1, 800000)

#s = avg_pool(signal[88], n_part=10000)
#del signal
#time = np.arange(0, 10000) + 1
#scales = np.arange(32, 256)
#plot_wavelet(time, s, scales, waveletname='morl' )

#s = medfilt(signal[0][:], kernel_size=13)

img_bgn = 0
img_end = 127
img_size = img_end - img_bgn
scales = np.arange(img_bgn + 1, img_end+1)
wav_name = 'morl'
X_train, X_test, y_train, y_test = train_test_split(signal, Y_data, test_size=0.3, shuffle=True)
del signal
chnls = 9
x_train = np.ndarray(shape=(len(X_train), img_size, img_size, chnls))
x_test = np.ndarray(shape= (len(X_test), img_size, img_size, chnls))

print("train: ")
for i in tqdm(range(len(X_train))):
    for j in range(0, chnls):

        s = avg_pool(X_train[i], n_part=10000)
        coff, freq = pywt.cwt(s, scales, wav_name, 1)
        image = coff[:img_size, :img_size]
        x_train[i, :, :, j] = image

del X_train

print("test: ")
for i in tqdm(range(len(X_test))):
    for j in range(0, chnls):

        s = avg_pool(X_test[i], n_part=10000)
        coff, freq = pywt.cwt(s, scales, wav_name, 1)
        image = coff[:img_size, :img_size]
        x_test[i, :, :, j] = image
print("done!")

del X_test

img_x = img_size
img_y = img_size
img_z = chnls
input_shape = (img_x, img_y, img_z)
 
batch_size = 1
num_classes = 1
epochs = 10
 
x_train = x_train.reshape(x_train.shape[0], img_x, img_y, img_z)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, img_z)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
 
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
 
model.compile(loss='binary_crossentropy',
              optimizer='adagrad',
              metrics=[matthews_correlation])
 
 
#class AccuracyHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.acc = []
# 
#    def on_epoch_end(self, batch, logs={}):
#        self.acc.append(logs.get('acc'))
# 
#history = AccuracyHistory()
 
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=True)
score = model.evaluate(x_test, y_test, verbose=False)
pred = model.predict(x_test)
pred = pred.reshape(-1)
pred = (pred > 0.7) * 1
mc = matthews_corrcoef(y_test, pred)
print ("mc: " , mc)
