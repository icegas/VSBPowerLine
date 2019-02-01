import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def main():
    signals = pq.read_table('data/train.parquet', columns=[str(i) for i in range(999)]).to_pandas()
    signals = np.array(signals).T.reshape((999//3, 3, 800000))

    plt.figure(figsize=(15, 10))
    plt.title("0")
    plt.plot(signals[0, 0, :], label='Phase 0')
   # plt.plot(signals[0, 1, :], label='Phase 1')
   # plt.plot(signals[0, 2, :], label='Phase 2')
    plt.legend()

    plt.figure(figsize=(15, 10))
    plt.title("190")
    mf = medfilt(signals[0, 0, :], kernel_size=13)
    plt.plot(mf)
   # plt.plot(signals[159, 0, :], label='Phase 0')
   # plt.plot(signals[159, 1, :], label='Phase 1')
   # plt.plot(signals[159, 2, :], label='Phase 2')
    plt.legend()

    #plt.figure(figsize=(15, 10))
    #plt.title("201")
    #plt.plot(signals[201, 0, :], label='Phase 0')
    #plt.plot(signals[201, 1, :], label='Phase 1')
    #plt.plot(signals[201, 2, :], label='Phase 2')
    #plt.legend()

    #plt.figure(figsize=(15, 10))
    #plt.title("202")
    #plt.plot(signals[202, 0, :], label='Phase 0')
    #plt.plot(signals[202, 1, :], label='Phase 1')
    #plt.plot(signals[202, 2, :], label='Phase 2')
    #plt.legend()

    plt.show()


if __name__ == "__main__":
    main()