import numpy as np
import pandas as pd
import os

def main():
    script_dir = os.path.dirname(__file__)
    proj_dir = os.path.abspath(os.path.join(script_dir, '../..'))
    data_dir = os.path.join(proj_dir, 'data/interim')
    out_dir = os.path.join(proj_dir, 'data/processed')

    # test_path = os.path.join(proj_dir, 'data/interim/test.csv')
    train_path = os.path.join(proj_dir, 'data/interim/train.csv')
    cv_path = os.path.join(proj_dir, 'data/interim/cv.csv')
    
    # Read in raw train and CV data
    # test_df = pd.read_csv(test_path)
    train_df = pd.read_csv(train_path)
    cv_df = pd.read_csv(cv_path)

    # Prepare features
    # test_df = prep_features(test_df)
    train_df = prep_features(train_df)
    cv_df = prep_features(cv_df)

    # Prepare response variable
    train_df = prep_response_var(train_df)
    cv_df = prep_response_var(cv_df)

    # Write out
    # test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    cv_df.to_csv(os.path.join(out_dir, 'cv.csv'), index=False)
    
def prep_features(df):
    # Nothing to do yet
    return df

def prep_response_var(df):
    # Log-transform the response variable, SalePrice
    df['SalePrice'] = df['SalePrice'].apply(np.log)
    return df

if __name__ == '__main__':
    main()
