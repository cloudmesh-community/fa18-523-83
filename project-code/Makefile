.PHONY: update-environment

update-environment:
	sudo apt-get update
	sudo apt-get -y upgrade

.PHONY: kaggleapi-install		
kaggleapi-install:
	pip3 install --user kaggle-api

.PHONY: packages-install		
packages-install:
	sudo pip3 install click
	sudo pip3 install pandas
	sudo pip3 install jupyter
	sudo pip3 install sklearn

download-file:
	kaggle datasets download -d brycecf/give-me-some-credit-dataset
	unzip -o give-me-some-credit-dataset.zip -d data/raw
	rm -f give-me-some-credit-dataset.zip


preprocessing-files:
	python3 src/data/preprocessing.py data/raw/cs-training.csv data/processed/training.pkl
	python3 src/data/preprocessing.py data/raw/cs-test.csv data/processed/testing.pkl



prepare-test-file: data/interim/testing.pkl
	python3 src/data/to_json.py data/interim/testing.pkl data/interim/test.json 


.PHONY: clean
clean:
	rm -f data/raw/*.csv
	rm -f data/raw/*.xls
	rm -f data/processed/*.pkl
	rm -f data/interim/*.json
