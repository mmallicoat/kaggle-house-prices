import pandas as pd
import os
import sys

def main(argv):
    # file paths relative to current directory
    infile = os.path.abspath(argv[1])
    outfile = os.path.abspath(argv[2])
    
    # script_dir = os.path.dirname(__file__)
    # proj_dir = os.path.abspath(os.path.join(script_dir, '../..'))
    # data_dir = os.path.join(proj_dir, 'data/interim')
    # out_dir = os.path.join(proj_dir, 'data/processed')
    # test_path = os.path.join(proj_dir, 'data/interim/test.csv')
    # train_path = os.path.join(proj_dir, 'data/interim/train.csv')
    # cv_path = os.path.join(proj_dir, 'data/interim/cv.csv')
    
    # Read in raw train, CV, or test data
    df = pd.read_csv(infile)
    # Prepare features
    df = prep_features(df)
    # Write out
    df.to_csv(outfile, index=False)
    
def prep_features(df):
    # Nothing to do yet
    return df

if __name__ == '__main__':
    main(sys.argv)
