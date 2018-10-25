import pandas as pd
import numpy as np
import os
import sys
from sklearn import preprocessing

def main(argv):
    # file paths relative to current directory
    infile = os.path.abspath(argv[1])
    outfile = os.path.abspath(argv[2])
    
    # Read in raw train, CV, or test data
    df = pd.read_csv(infile)

    # Handle NA's
    fields = df.columns.tolist()
    for field in fields:
        if (df.dtypes[field] == np.dtype('float64')
            or df.dtypes[field] == np.dtype('int64')):  # numeric
            df[field].fillna(np.mean(df[field]), inplace=True)
        elif df.dtypes[field] == np.dtype('O'):  # categorical/ordinal
            df[field].fillna('Unknown', inplace=True)
        else:  # unknown
            df.drop(field, axis=1, inplace=True)
            fields.remove(field)
    
    # Encode categorical variables
    enc = preprocessing.OneHotEncoder(sparse=False)
    n = len(df)
    for field in fields:
        if df.dtypes[field] == np.dtype('O'):  # categorical/ordinal
            arr = df[field].values.reshape(n,1)
            enc.fit(arr)
            arr = enc.transform(arr)
            # Replace with encoded variables in dataframe
            df.drop(field, axis=1, inplace=True)
            for i in range(arr.shape[1]):
                df[field + '_%s' % i] = arr[:,i]

    # Prepare features
    # df = prep_features(df)
    # NOTE: feature selection now in train_model.py
    # If we engineer any features, code will be here

    # Write out
    df.to_csv(outfile, index=False)
    
if __name__ == '__main__':
    main(sys.argv)
