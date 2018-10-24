import numpy as np
import pandas as pd
from sklearn import linear_model
import os
import sys
import pickle

def main(argv):
    trainfile = os.path.abspath(argv[1])
    modelfile = os.path.abspath(argv[2])

    # Read in data and store in arrays
    X, y = csv_to_vars(trainfile)
    X, y = format_data(X, y)

    # Train
    model = train(X, y)

    # Serialize model to pickle file
    pickle.dump(model, open(modelfile, 'wb'))

def format_data(X, y):
    X = X.values
    if y is not None:
        y = y.values  # convert to numpy array
        y = y.reshape(y.shape[0], 1)  # reshape
        # TODO: leave log-transform to the GLM model
        y = np.log(y)  # log-transform
    return X, y

# reads in csv; splits indep and dep variables
def csv_to_vars(filepath):
    df = pd.read_csv(filepath)
    if 'SalePrice' in df.columns.tolist():
        dep_vars = df.columns.tolist()
        dep_vars.remove('SalePrice')
        y = df['SalePrice']
        X = df[dep_vars]
    else:  # no response variable
        X = df
        y = None
    return X, y

def train(X, y):
    # Normalize subtracts mean from each variable?
    # X array does not have an intercept
    model = linear_model.LinearRegression(fit_intercept=True,
                                          normalize=True)
    model.fit(X, y)
    return model

if __name__ == '__main__':
    main(sys.argv)

