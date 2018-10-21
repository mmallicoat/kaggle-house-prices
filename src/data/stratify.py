import pandas as pd
import random
import os
import sys

def main(argv):
    infile = os.path.abspath(argv[1])
    outdir = os.path.abspath(argv[2])

    # script_dir = os.path.dirname(__file__)
    # proj_dir = os.path.abspath(os.path.join(script_dir, '../..'))
    # total_train_path = os.path.join(proj_dir, 'data/raw/train.csv')
    # train_path = os.path.join(proj_dir, 'data/interim/train.csv')
    # cv_path = os.path.join(proj_dir, 'data/interim/cv.csv')

    # Stratify raw training data in train and CV datasets
    total_train = pd.read_csv(infile)

    # Calculate number of observations to keep in each dataset
    n = len(total_train)
    n_cv = int(.25 * n)
    n_train = n - n_cv
    random.seed(666)  # set random seed
    cv_indices = random.sample(total_train.index, n_cv)
    cv = total_train.ix[cv_indices]
    train = total_train.drop(cv_indices)

    # Write out
    cv.to_csv(os.path.join(outdir, 'cv.csv'), index=False)
    train.to_csv(os.path.join(outdir, 'train.csv'), index=False)

if __name__ == '__main__':
    main(sys.argv)
