import numpy as np
import matplotlib.pyplot as plt
import pywt
import pyarrow.parquet as pq
from scipy import signal

#def filter_signal(signal, threshold=10):
#    fourier = rfft(signal)
#    frequencies = rfftfreq(signal.size, d=2e-1/signal.size)
#    fourier[frequencies > threshold] = 0
#    return irfft(fourier)

def apply_convolution(sig, window=1):
    """Apply a simple same-size convolution with a given window size"""

    a = np.linspace(-2,2,25)
    conv = np.array([((i*i))*np.sqrt(1/np.pi)*np.exp(-(i*i)/2) for i in a])
    #conv = np.repeat([0., 1., 2., 1., 0], window)
    filtered = signal.convolve(sig, conv, mode='same') / window
    return filtered

def maddest(d, axis=None):
    """
    Mean Absolute Deviation
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def waveletSmooth( x, level, wavelet="db4", title=None ):
    # calculate the wavelet coefficients
    coeff = pywt.wavedec( x, wavelet, mode="per" )
    # calculate a threshold
    sigma = maddest( coeff[-level] )
    # changing this threshold also changes the behavior,
    # but I have not played with this very much
    uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
    coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="soft" ) for i in coeff[1:] )
    # reconstruct the signal using the thresholded coefficients
    y = pywt.waverec( coeff, wavelet, mode="per" )

    return y

def filter_signal(signal, threshold):
    fourier = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=20e-3/signal.size) #signal.size
    fourier[frequencies > threshold] = 0
    return np.fft.irfft(fourier)

def main():
    signals = pq.read_table('data/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
    signals = np.array(signals).T.reshape((999//3, 3, 800000))

    num = 190
    plt.figure(figsize=(15, 10))
    plt.title('original')
    plt.subplot(221)
    plt.plot(signals[num, 0, :], label='Phase 0')
    #plt.plot(signals[num, 1, :], label='Phase 1')
    #plt.plot(signals[num, 2, :], label='Phase 2')
    plt.legend()

    plt.title('1')
    graphs = [222, 223, 224]

    arr = np.array([])

    for i in range(3):
        arr = np.append(arr, waveletSmooth(signals[num, 0, :], i , wavelet='db4') )#, label='Phase 0')
        arr = np.append(arr, waveletSmooth(signals[num, 1, :], i , wavelet='db4') )#, label='Phase 1')
        arr = np.append(arr, waveletSmooth(signals[num, 2, :], i , wavelet='db4') )#, label='Phase 2')

    arr = np.reshape(arr, (3, 3, -1))
    plt.subplot(graphs[0])
    plt.plot(arr[1][0], label = 'Phase 0')
    plt.subplot(graphs[1])
    plt.plot(filter_signal(arr[1][0], 1e4), label = 'Phase 0')

    #for i in range(len(arr)):
    #    plt.subplot(graphs[i])
    #    plt.plot(peaks_remove(arr[i][0]), label = 'Phase 0')
    #    plt.plot(peaks_remove(arr[i][1]), label = 'Phase 1')
    #    plt.plot(peaks_remove(arr[i][2]), label = 'Phase 2')


    plt.show()

if __name__ == '__main__':
    main()