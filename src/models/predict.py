import numpy as np
import pandas as pd
from sklearn import linear_model
import sys
import os
import train_model.py
import pickle

# TODO: this entire file needs to be rewritten. This should take a
# pretrained model, either the parameters themselves or the filepath
# of the parameters, and return the predictions or write out the
# predictions to a supplied location.
# This should perhaps also calculate the error, too, if a flag is
# set and the true values of response variable are supplied.
# For the test dataset, error will not be possible to calculate.

def main(argv):
    modelfile = os.path.abspath(argv[1])
    datafile = os.path.abspath(argv[2])
    predictfile = os.path.abspath(argv[3])
    calc_error = os.path.abspath(argv[4])

    # Read in model
    model = pickle.load(open(modelfile, 'rb'))

    # Read in data to predict, store in arrays
    # TODO: check if this gives error if no response var
    X, y = train_model.csv_to_vars(datafile)
    # TODO: Repeat feature selection as in model definition

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error
    if calc_error:
        # TODO: shapes may not match here
        error = loss_function(y_pred, y)
        print("RMSE is %f: " % error)

    # Output test predictions
    # TODO: check syntax
    # NOTE: response variable log-transform is NOT reversed here
    indices = range(1461, 1461 + len(y_pred))
    y_pred = pd.DataFrame(y_pred, index=indices, columns=['SalePrice'])
    y_pred.to_csv(predictfile, index_label='Id')
    
def loss_function(y_pred, y):  # 1-dim numpy arrays of equal length
    SSE = float(sum((y - y_pred) ** 2))  # must be float to divide
    RMSE = (SSE / len(y)) ** .5
    return RMSE

if __name__ == '__main__':
    main(sys.argv)
