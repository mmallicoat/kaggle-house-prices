import pandas as pd
import numpy as np
import os
import sys
from sklearn import preprocessing
import pdb

def main(argv):
    # file paths relative to current directory
    inpath = os.path.abspath(argv[1])
    outpath = os.path.abspath(argv[2])
    
    # Read in raw train, CV, or test data
    df_train = pd.read_csv(os.path.join(inpath, 'train.csv'))
    df_test = pd.read_csv(os.path.join(inpath, 'test.csv'))
    df_cv = pd.read_csv(os.path.join(inpath, 'cv.csv'))

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
    entropy_df.sort_values(by='Entropy', inplace=True)
    # Select variables with Shannon entropy less than 0.5
    select_cat_vars = entropy_df[entropy_df['Entropy'] < .5].index.tolist()

    # Select numeric variables
    # TODO: calculate variance of numeric variables

    # Remove non-selected variables
    # TODO
    
    # Encode categorical variables
    # TODO: rewrite this encapsulating in a function
    for field in fields:
        if df_train.dtypes[field] == np.dtype('O'):
            ar_train = df_train[field].values.reshape(len(df_train), 1)
            ar_test = df_test[field].values.reshape(len(df_test), 1)
            ar_cv = df_cv[field].values.reshape(len(df_cv), 1)

            # Build list of unqiue values for field
            values = df_train[field].unique()
            values = np.append(values, df_test[field].unique())
            values = np.append(values, df_cv[field].unique())
            values = [np.unique(values)]  # list of arrays
            
            # Transform
            encoder = preprocessing.OneHotEncoder(
                                            categories=values,
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
    df_train.to_csv(os.path.join(outpath, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(outpath, 'test.csv'), index=False)
    df_cv.to_csv(os.path.join(outpath, 'cv.csv'), index=False)

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
