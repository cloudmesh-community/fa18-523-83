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
download-file:
	kaggle datasets download -d brycecf/give-me-some-credit-dataset
	unzip -o give-me-some-credit-dataset.zip -d data/raw
	rm -f give-me-some-credit-dataset.zip


preprocessing-training-file:
	python3 src/data/preprocessing.py 'data/raw/cs-training.csv' 'data/interim/training.pkl'


.PHONY: clean
clean:
	rm -f data/raw/*.csv
	rm -f data/raw/*.xls
