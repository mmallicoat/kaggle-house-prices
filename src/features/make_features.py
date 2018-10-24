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
    df = prep_features(df)
    # Write out
    df.to_csv(outfile, index=False)
    
def prep_features(df):
    # Select features to use in models
    features = ['LotArea', 'YearBuilt']
    # Keep independent var if available
    columns = df.columns.tolist()
    if 'SalePrice' in columns:
        features.append('SalePrice')
    df = df[features]
    return df

if __name__ == '__main__':
    main(sys.argv)
