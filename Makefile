interim: data/interim/train.csv data/interim/cv.csv data/interim/test.csv

processed: data/processed/train.csv data/processed/test.csv data/processed/cv.csv

# Stratify training data and copy test data forward
data/interim/train.csv data/interim/cv.csv:
	python src/data/stratify.py data/raw/train.csv data/interim
    
data/interim/test.csv:
	cp data/raw/test.csv data/interim/test.csv

# Make features
data/processed/train.csv: data/interim/train.csv
	python src/features/make_features.py data/interim/train.csv data/processed/train.csv

data/processed/cv.csv: data/interim/cv.csv
	python src/features/make_features.py data/interim/cv.csv data/processed/cv.csv

data/processed/test.csv: data/interim/test.csv
	python src/features/make_features.py data/interim/test.csv data/processed/test.csv
