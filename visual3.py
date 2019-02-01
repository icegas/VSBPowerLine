import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import seaborn as sns
from string import ascii_letters

def main():

    sns.set(style="white")

    size = 999

    signals = pq.read_table('data/train.parquet', columns=[str(i) for i in range(size)]).to_pandas()
    signals = np.array(signals).T.reshape((size//3, 3, 800000))

    corr = np.corrcoef(signals[:30,0,:])
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    #f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="dark")
    #(220, 10, as_cmap=True)

    sns.heatmap(corr)#, mask=mask, cmap=cmap, vmax=.3, center=0,
            #square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.show()

if __name__=='__main__':
    main()
