import pandas as pd
import os
import sys

def main(argv):
    # file paths relative to current directory
    infile = os.path.abspath(argv[1])
    outfile = os.path.abspath(argv[2])
    
    # Read in raw train, CV, or test data
    df = pd.read_csv(infile)

    # Prepare features
    # df = prep_features(df)
    # NOTE: feature selection now in train_model.py
    # If we engineer any features, code will be here

    # Write out
    df.to_csv(outfile, index=False)
    
if __name__ == '__main__':
    main(sys.argv)
