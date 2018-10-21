interim: data/interim/train.csv data/interim/test.csv data/interim/cv.csv

processed: data/processed/train.csv data/processed/test.csv data/processed/cv.csv

data/interim/train.csv:
	python src/data/stratify.py
    
data/interim/cv.csv:
	python src/data/stratify.py

data/interim/test.csv:
	cp data/raw/test.csv data/interim/test.csv

data/processed/train.csv: data/interim/train.csv
	python src/features/make_features.py

data/processed/cv.csv: data/interim/cv.csv
	python src/features/make_features.py

data/processed/test.csv: data/interim/test.csv
	cp data/interim/test.csv data/processed/test.csv
