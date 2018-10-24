import numpy as np
import pandas as pd
from sklearn import linear_model
import sys
import os
import pickle
import train_model  # local module

def main(argv):
    modelfile = os.path.abspath(argv[1])
    datafile = os.path.abspath(argv[2])
    predictfile = os.path.abspath(argv[3])
    # calc_error = os.path.abspath(argv[4])

    # Read in model
    model = pickle.load(open(modelfile, 'rb'))

    # Read in data to predict, store in arrays
    X, y = train_model.csv_to_vars(datafile)
    X, y = train_model.format_data(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error
    if y is not None:
        error = loss_function(y_pred, y)
        print("RMSE is %f" % error)

    # Output test predictions
    # NOTE: move log-transform to GLM model
    if y is None:  # set proper Id for submission
        indices = range(1461, 1461 + len(y_pred))
    else:
        indices = range(1, 1 + len(y_pred))
        y_pred = np.exp(y_pred)  # reverse log-transofrm
        y_pred = pd.DataFrame(y_pred, index=indices, columns=['SalePrice'])
        y_pred.to_csv(predictfile, index_label='Id')
    
def loss_function(y_pred, y):  # 1-dim numpy arrays of equal length
    SSE = float(sum((y - y_pred) ** 2))  # must be float to divide
    RMSE = (SSE / len(y)) ** .5
    return RMSE

if __name__ == '__main__':
    main(sys.argv)
