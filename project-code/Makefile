.PHONY: update-environment
update-environment:
	sudo apt-get update
	sudo apt-get -y upgrade


.PHONY: docker-install
docker-install:
	sudo apt-get install apt-transport-https ca-certificates curl software-properties-common
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
	sudo apt-key fingerprint 0EBFCD88
	sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"
	sudo apt-get update
	sudo apt-get install docker-ce

.PHONY: docker-build-image
docker-build-image:
	cd src/docker ; sudo docker build -t test_app . ; cd ../../

.PHONY: docker-run-image
docker-run-image:
	sudo docker run -p 4000:80 test_app 

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
	sudo apt-get install -y unzip zip ; kaggle datasets download -d brycecf/give-me-some-credit-dataset ; unzip -o give-me-some-credit-dataset.zip -d data/raw ; rm -f give-me-some-credit-dataset.zip

preprocessing-files:
	python3 src/data/preprocessing.py data/raw/cs-training.csv data/processed/training.pkl
	python3 src/data/preprocessing.py data/raw/cs-test.csv data/interim/testing.pkl

prepare-test-file: data/interim/testing.pkl
	python3 src/data/to_json.py data/interim/testing.pkl data/interim/test.json 

create-model:
	python3 src/data/train_data.py data/processed/training.pkl src/docker/xgboost_model.pkl

.PHONY: clean
clean:
	rm -f data/raw/*.csv
	rm -f data/raw/*.xls
	rm -f data/processed/*.pkl
	rm -f data/interim/*.pkl
	rm -f data/processed/*.json
	rm -f src/model/*.pkl

.PHONY: post-test-data
post-test-data: 
	curl -i -X POST -H "Content-Type: application/json" -d @data/interim/test.json http://0.0.0.0:4000/json-get-results

.PHONE: docker-clean
docker-clean:
	id= echo `sudo docker ps -q` ; sudo docker kill ${id}
