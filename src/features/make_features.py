import pandas as pd
import numpy as np
import os
import sys
from sklearn import preprocessing

def main(argv):
    # file paths relative to current directory
    inpath = os.path.abspath(argv[1])
    outpath = os.path.abspath(argv[2])
    
    # Read in raw train, CV, or test data
    df_train = pd.read_csv(os.path.join(inpath, 'train.csv'))
    df_test = pd.read_csv(os.path.join(inpath, 'test.csv'))
    df_cv = pd.read_csv(os.path.join(inpath, 'cv.csv'))

    # Handle NA's
    fields = df_train.columns.tolist()
    fields.remove('SalePrice')  # ignore response variable
    for field in fields:
        if (df_train.dtypes[field] == np.dtype('float64')
            or df_train.dtypes[field] == np.dtype('int64')):  # numeric
            mean = np.mean(df_train[field])
            df_train[field].fillna(mean, inplace=True)
            df_test[field].fillna(mean, inplace=True)
            df_cv[field].fillna(mean, inplace=True)
        elif df_train.dtypes[field] == np.dtype('O'):  # string
            df_train[field].fillna('Unknown', inplace=True)
            df_test[field].fillna('Unknown', inplace=True)
            df_cv[field].fillna('Unknown', inplace=True)
        else:  # unknown
            df_train.drop(field, axis=1, inplace=True)
            df_test.drop(field, axis=1, inplace=True)
            df_cv.drop(field, axis=1, inplace=True)
            fields.remove(field)
    
    # Encode categorical variables
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

    # Prepare features
    # NOTE: feature selection is done in train_model.py
    # But if we engineer any features, code will be here

    # Write out
    df_train.to_csv(os.path.join(outpath, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(outpath, 'test.csv'), index=False)
    df_cv.to_csv(os.path.join(outpath, 'cv.csv'), index=False)

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
