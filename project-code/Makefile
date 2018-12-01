.PHONY: update-environment

update-environment:
	sudo apt-get update
	sudo apt-get -y upgrade

	

.PHONY: kaggleapi-install		
kaggleapi-install:
	pip3 install --user kaggle-api


docker-install-before-ubuntu18-10:
	sudo apt-get update
	sudo apt-get install \ apt-transport-https \ ca-certificates \ curl \ software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo apt-key fingerprint 0EBFCD88
	sudo add-apt-repository \ "deb [arch=amd64] https://download.docker.com/linux/ubuntu \ (lsb_release -cs) \ stable"
	sudo apt-get update
	sudo apt-get install docker-ce

docker-install-ubuntu18-10:
	sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo apt-key fingerprint 0EBFCD88
	sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
	sudo apt-get update
	sudo apt-get install docker-ce


docker-start:
	sudo docker start test_app


docker-stop:
	sudo docker stop test_app


docker-build-image:
	sudo docker build -t test_app .

docker-run-image:
	cd src/docker 
	sudo docker run -p 4000:80 test_app 
	cd ../../

	
.PHONY: packages-install		
packages-install:
	sudo apt install python3-pip
	sudo pip3 install click
	sudo pip3 install pandas
	sudo pip3 install jupyter
	sudo pip3 install sklearn
	sudo pip3 install xgboost
	sudo pip3 install flask
	sudo pip3 install kaggle

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

.PHONY: post-test-data
post-test-data: 
	curl -X localhost:80/json-get-results -d @data/interim/test.json --header 'Content-Type: application/json'
