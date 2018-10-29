import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
#from sklearn_pandas import DataFrameMapper
import os
import sys
import pickle
import pdb

def main(argv):
    trainfile = os.path.abspath(argv[1])
    modelstem = os.path.abspath(argv[2])  # path and basename, no ext

    # Read and prepare data
    train_df = pd.read_csv(trainfile, index_col=0)
    X, y = prep_data(train_df)

    # Scale and transform
    X, scaler = scale_dep_vars(X)
    y = transform_response(y)

    # Train model
    model = train(X, y)

    # Serialize model params to pickle file
    pickle.dump(model, open(modelstem + '.p', 'wb'))
    scaler_params = np.array([scaler.mean_, scaler.scale_])  # mean, stdev
    np.save(modelstem + '-scaler.npy', scaler_params)

def scale_dep_vars(X):
    # Fit scaler and standardize dependent vars
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # Transformation can be reversed with:
    # X = X * scaler.scale_ + scaler.mean_
    return X, scaler

def transform_response(y):
    # Tranform response variable
    # TODO: fit a Box-Cox transformer for y
    y = np.log(y)  # log-transform
    return y
    
def prep_data(df):
    # From dataframe, select features and return numpy array.
    # NOTE: Must allow for case of no 'SalePrice' (response variable)
    # field, since this function is called in predict.py.

    # Split indep and dep variables
    if 'SalePrice' in df.columns.tolist():
        dep_vars = df.columns.tolist()
        dep_vars.remove('SalePrice')
        X = df[dep_vars]
        y = df['SalePrice']
    else:  # no response variable
        X = df
        y = None

    # NOTE: Feature selection is done in make_features.py

    # Convert to numpy arrays
    X = X.values.astype(np.float64)
    if y is not None:
        y = y.values.astype(np.float64)
        y = y.reshape(y.shape[0], 1)  # reshape
    return X, y


def train(X, y):
    # X array does not have an intercept, so set fit_intercept
    # Since standardization already perform, do not normalize
    model = linear_model.Ridge(alpha=1, fit_intercept=True,
                               normalize=False)
    model.fit(X, y)
    return model

if __name__ == '__main__':
    main(sys.argv)

