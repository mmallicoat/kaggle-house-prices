import numpy as np
import pandas as pd
import sys
import os
import pickle
import train_model  # local module
import pdb

def main(argv):
    modelstem = os.path.abspath(argv[1])
    datafile = os.path.abspath(argv[2])
    predictfile = os.path.abspath(argv[3])

    # Read in model and other params
    model = pickle.load(open(modelstem + '.p', 'rb'))
    scaler_params = np.load(modelstem + '-scaler.npy')

    # Read in data to predict, store in arrays
    df = pd.read_csv(datafile)
    X, y = train_model.prep_data(df)

    # Scale and transform variables
    X = (X - scaler_params[0]) / scaler_params[1]

    # Make predictions
    y_pred = model.predict(X)
    # Reverse response transform
    y_pred = np.exp(y_pred)

    # Calculate error of untransformed response var
    if y is not None:
        error = loss_function(y_pred, y)
        print("RMSE is %f" % error)

    # Output test predictions
    if y is None:  # set proper Id for submission
        indices = range(1461, 1461 + len(y_pred))
    else:
        indices = range(1, 1 + len(y_pred))
    y_pred = pd.DataFrame(y_pred, index=indices, columns=['SalePrice'])
    y_pred.to_csv(predictfile, index_label='Id')
    
def loss_function(y_pred, y):  # 1-dim numpy arrays of equal length
    SSE = float(sum((y - y_pred) ** 2))  # must be float to later divide
    RMSE = (SSE / len(y)) ** .5
    return RMSE

if __name__ == '__main__':
    main(sys.argv)
