import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import os
import pdb

def main(argv):
    # file paths relative to current directory
    inpath = os.path.abspath(argv[1])
    outpath = os.path.abspath(argv[2])
    
    # Read in raw train, CV, or test data
    df_train = pd.read_csv(os.path.join(inpath, 'train.csv'), index_col=0)
    df_test = pd.read_csv(os.path.join(inpath, 'test.csv'), index_col=0)
    df_cv = pd.read_csv(os.path.join(inpath, 'cv.csv'), index_col=0)

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
    df_test = handle_nan(sub, df_test)
    df_cv = handle_nan(sub, df_cv)

    # Engineer features
    # None for now.

    # Select categorical variables
    entropy_dict = dict()
    for var in cat_vars:
        entropy_dict[var] = entropy(df_train[var])
    entropy_df = pd.DataFrame.from_dict(entropy_dict,
                                        orient='index',
                                        columns=['Entropy'])
    # Select variables with Shannon entropy greater than threshold
    select_cat_vars = entropy_df[entropy_df['Entropy'] > 1.4].index.tolist()

    # Select numeric variables
    var_dict = dict()
    for var in num_vars:
        var_dict[var] = np.var(df_train[var])
    var_df = pd.DataFrame.from_dict(var_dict, orient='index',
                                    columns=['Variance'])
    # var_df.sort_values(by='Variance', ascending=False, inplace=True)
    # Select variables with variance greater than threshold
    select_num_vars = var_df[var_df['Variance'] > 10 ** 4].index.tolist()
    try:
        select_num_vars.remove('SalePrice')  # don't include response variable
    except:
        pass

    # Remove non-selected variables
    select_vars = select_num_vars + select_cat_vars
    df_train = df_train[select_vars + ['SalePrice']]  # keep response var
    df_cv = df_cv[select_vars + ['SalePrice']]
    df_test = df_test[select_vars]
    
    # Encode categorical variables
    # TODO: rewrite this encapsulating in a function
    for field in select_cat_vars:
        ar_train = df_train[field].values.reshape(len(df_train), 1)
        ar_test = df_test[field].values.reshape(len(df_test), 1)
        ar_cv = df_cv[field].values.reshape(len(df_cv), 1)

        # Build list of unqiue values for field
        values = df_train[field].unique()
        values = np.append(values, df_test[field].unique())
        values = np.append(values, df_cv[field].unique())
        values = [np.unique(values)]  # list of arrays
        
        # Transform
        encoder = preprocessing.OneHotEncoder(categories=values,
                                              sparse=False)
        encoder.fit(ar_train)
        ar_train = encoder.transform(ar_train)
        ar_test = encoder.transform(ar_test)
        ar_cv = encoder.transform(ar_cv)

        # Replace with encoded variables in dataframe
        df_train.drop(field, axis=1, inplace=True)
        df_test.drop(field, axis=1, inplace=True)
        df_cv.drop(field, axis=1, inplace=True)
        for i in range(ar_train.shape[1]):
            df_train[field + '_%s' % i] = ar_train[:,i]
            df_test[field + '_%s' % i] = ar_test[:,i]
            df_cv[field + '_%s' % i] = ar_cv[:,i]

    # Write out
    df_train.to_csv(os.path.join(outpath, 'train.csv'))
    df_test.to_csv(os.path.join(outpath, 'test.csv'))
    df_cv.to_csv(os.path.join(outpath, 'cv.csv'))

def histogram(x):  # x is a Pandas Series
    # mpl.use('TkAgg')  # needed sometimes for obscure reasons
    plt.hist(x, bins=10)
    # plt.xticks(np.arange(min(x), max(x)+1, .5))
    # plt.axis([min(x), 10**6, 0, 200])
    plt.grid(True)
    plt.show()

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
