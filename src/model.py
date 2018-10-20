import numpy as np
import pandas as pd
from sklearn import linear_model
# import matplotlib.pyplot as plt
import random


def main():
    # Read in data and store in arrays
    data_path = '~/Documents/Data Science/Kaggle House Prices/Data/'
    sub_path = '~/Documents/Data Science/Kaggle House Prices/Submission/'
    X, y = csv_to_df(data_path + 'train_excl_cv.csv')
    X_cv, y_cv = csv_to_df(data_path + 'cv.csv')
    features = ['LotArea', 'YearBuilt']
    X = prep_features(X, features)
    X_cv = prep_features(X_cv, features)
    y = prep_dep_var(y)
    y_cv = prep_dep_var(y_cv)

    # Train
    reg = train(X, y)

    # Predict
    cv_pred = predict(reg, X_cv)

    # Calculate error
    error = loss_function(y_cv, cv_pred)
    print(error)

    # Output test predictions
    if False:
        output_predictions(cv_pred, sub_path + 'model2.csv')


def output_predictions(y_pred, filepath):
    # Indices of test data for submission will be 1461 through 2919
    indices = range(1461, 1461 + len(y_pred))
    y_pred = pd.DataFrame(np.exp(y_pred),
                          index=indices,
                          columns=['SalePrice'])
    y_pred.to_csv(filepath, index_label='Id')


def csv_to_df(filepath):  # reads csv into pandas dataframe
    df = pd.read_csv(filepath)
    features = df.columns.tolist()
    features.remove('SalePrice')
    y = df['SalePrice']
    X = df[features]
    return X, y


def prep_dep_var(y):  # pandas dataframe
    y = y.values  # return numpy array
    y = y.reshape(y.shape[0], 1)
    y = np.log(y)
    return y


def prep_features(X, feature_list):  # pandas dataframe and list
    X = X[feature_list].values  # returns numpy array
    return X


def loss_function(y, y_pred):  # 1-dim numpy arrays of equal length
    SSE = float(sum((y - y_pred) ** 2))
    RMSE = (SSE / len(y)) ** .5  # need to make sure float before division
    return RMSE


def predict(model, X):
    pred = model.predict(X)
    return pred


def train(X, y):
    model = linear_model.LinearRegression(fit_intercept=True,
                                          normalize=True)
    model.fit(X, y)
    return model


def plots():
    # Plot learning curve
    # TODO: write code to plot this

    # Plots
    # plot it as in the example at http://scikit-learn.org/
    # plt.scatter(X, y,  color='black')
    # plt.plot(X, reg.predict(X), color='blue', linewidth=3)
    # plt.xticks(())
    # plt.yticks(())
    # plt.show()
    return


def prepocess():
    # TODO: change function to take train dataframe as an argument
    # then return the split dataframes
    # Split training dataset into train and CV
    train_full = pd.read_csv('./Data/train.csv')
    cv_indices = random.sample(train_full.index, 460)
    cv = train_full.ix[cv_indices]
    train = train_full.drop(cv_indices)
    # Write out
    cv.to_csv(path_or_buf='./Data/cv.csv', sep=',')
    train.to_csv(path_or_buf='./Data/train_excl_cv.csv', sep=',')


# X is an 2-dim array without intercept
# y is an 1-dim array
# Returns linear model object
def fit_regression(X, y):
    # y = np.log(y)
    reg = linear_model.LinearRegression(fit_intercept=True,
                                        normalize=True)
    reg.fit(X, y)
    return reg


if __name__ == '__main__':
    main()
