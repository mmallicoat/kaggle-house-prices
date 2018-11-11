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
    df = pd.read_csv(datafile, index_col=0)
    X, y = train_model.prep_data(df)

    # Scale and transform variables
    X = (X - scaler_params[0]) / scaler_params[1]

    # Make predictions
    y_pred = model.predict(X)

    # Calculate error of log-transformed response var
    if y is not None:
        error = loss_function(np.exp(y_pred), y)
        print("RMSE of untransformed prices is %f" % error)
        error = loss_function(y_pred, np.log(y))
        print("RMSE of log-transformed prices is %f" % error)

    # Output test predictions
    y_pred = np.exp(y_pred)  # reverse transformation of response var
    y_pred_df = pd.DataFrame(y_pred, index=df.index, columns=['SalePrice'])
    y_pred_df.to_csv(predictfile)
    
def loss_function(y_pred, y):  # 1-dim numpy arrays of equal length
    SSE = float(sum((y - y_pred) ** 2))  # must be float to later divide
    RMSE = (SSE / len(y)) ** .5
    return RMSE

if __name__ == '__main__':
    main(sys.argv)
