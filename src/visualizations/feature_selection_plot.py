import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sys
import os
import pdb

# Most of this code copied from make_features.py
def main(argv):
    # file paths relative to current directory
    inpath = os.path.abspath(argv[1])
    outputdir = os.path.abspath(argv[2])
    
    # Read in raw train, CV, or test data
    df_train = pd.read_csv(os.path.join(inpath, 'train.csv'), index_col=0)

    # Get list of variables by data type
    var_types = df_train.dtypes  # Series
    fields = var_types.index.tolist()
    cat_vars = var_types[var_types == 'O'].index.tolist()
    num_vars = var_types[(var_types == 'int64') |
                         (var_types == 'float64')].index.tolist()
    # Check all variables accounted for
    assert len(var_types) == len(cat_vars) + len(num_vars)

    # Handle NA's
    # Compile the default values to substitute
    sub = dict()
    for var in cat_vars:
        sub[var] = 'Unknown'
    for var in num_vars:
        sub[var] = np.mean(df_train[var])
    # Make substituions
    df_train = handle_nan(sub, df_train)

    # Select categorical variables
    entropy_dict = dict()
    for var in cat_vars:
        entropy_dict[var] = entropy(df_train[var])
    entropy_df = pd.DataFrame.from_dict(entropy_dict,
                                        orient='index',
                                        columns=['Entropy'])
    th = 1.4

    # Plot histogram of entropy values
    plt.title("Shannon Entropy of Categorical Variables")
    plt.hist(entropy_df.values, bins=10)
    plt.grid(True)
    plt.axvline(x=th, color='r')
    plt.savefig(os.path.join(outputdir, 'categorical-selection.png'))
    plt.clf()

    # Select numeric variables
    var_dict = dict()
    for var in num_vars:
        var_dict[var] = np.var(df_train[var])
    var_df = pd.DataFrame.from_dict(var_dict, orient='index',
                                    columns=['Variance'])
    var_df.drop('SalePrice', axis=0, inplace=True)
    th = 10 ** 4

    # Plots of numeric variables
    # Scatter plot of variance
    plt.title("Variance of Numeric Variables")
    var_df.sort_values('Variance', inplace=True)
    var_df.drop('LotArea', axis=0, inplace=True)
    plt.scatter(np.arange(len(var_df)),
                var_df.values)
    plt.xlabel("Index of Variable")
    plt.axhline(y=th, color='r')
    plt.savefig(os.path.join(outputdir, 'numeric-selection.png'))
    plt.clf()


def handle_nan(sub_dict, df):
    fields = df.columns.tolist()
    for field in fields:
        df[field].fillna(sub_dict[field], inplace=True)
    return df


def entropy(array):  # numpy array
    # Calculate Shannon's entropy for array of strings
    items = np.unique(array)
    n = array.shape[0]
    freq = dict()
    for item in items:
        freq[item] = float(np.where(array == item)[0].shape[0]) / n
    sum = 0.
    for item in freq:
        sum -= freq[item] * np.log2(freq[item])
    return sum
    

# https://www.peterbe.com/plog/uniqifiers-benchmark
def make_unique(seq, idfun=None): 
   # order preserving
   if idfun is None:
       def idfun(x): return x
   seen = {}
   result = []
   for item in seq:
       marker = idfun(item)
       # in old Python versions:
       # if seen.has_key(marker)
       # but in new ones:
       if marker in seen: continue
       seen[marker] = 1
       result.append(item)
   return result

if __name__ == '__main__':
    main(sys.argv)
