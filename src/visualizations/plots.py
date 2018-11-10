import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import os
import sys
import pdb

def main(argv):
    datafile = os.path.abspath(argv[1])
    outputdir = os.path.abspath(argv[2])

    # Read in data to predict, store in arrays
    df = pd.read_csv(datafile, index_col=0)
    y = df['SalePrice']

    # Calculate error of log-transformed response var
    if y is not None:
        y_transform = np.log(y)

    # Plot untransformed response variable
    plt.hist(y, bins=30, density=True)
    plt.title("House Sale Prices")
    mu = np.mean(y)
    variance = np.var(y)
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.yticks([])  # hide y axis
    plt.savefig(os.path.join(outputdir, 'y-hist.png'))
    plt.clf()

    # Plot transformed response variable
    plt.hist(y_transform, bins=30, density=True)
    plt.title("Log-Transformed House Sale Prices")
    mu = np.mean(y_transform)
    variance = np.var(y_transform)
    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.yticks([])  # hide y axis
    plt.savefig(os.path.join(outputdir, 'y-transformed-hist.png'))


if __name__ == '__main__':
    main(sys.argv)

