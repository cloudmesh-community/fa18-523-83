APP_NAME=credit_app
SHELL = ./bin/report_time.sh

.PHONY: prepare_environment
prepare-environment: update-environment packages-install docker-install 

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

.PHONY: docker-build-image
docker-build-image:
	cd src/docker ; sudo docker build -t $(APP_NAME) . ; cd ../../ $@

.PHONY: docker-run-image
docker-run-image:
	sudo docker run -p 80:80  --name="$(APP_NAME)" $(APP_NAME)

.PHONY: docker-stop
docker-stop: 
	sudo docker stop $(APP_NAME); sudo docker rm $(APP_NAME)

file-prep: download-file preprocessing-files prepare-test-file create-model 


download-file:
	sudo apt-get install -y unzip zip ; sudo kaggle datasets download -d brycecf/give-me-some-credit-dataset ; sudo unzip -o give-me-some-credit-dataset.zip -d data/raw ; sudo rm -f give-me-some-credit-dataset.zip

prepare-files: preprocessing-files prepare-test-file

preprocessing-files:
	sudo python3 src/data/preprocessing.py data/raw/cs-training.csv data/processed/training.pkl
	sudo python3 src/data/preprocessing.py data/raw/cs-test.csv data/interim/testing.pkl
	$@
	
prepare-test-file: 
	sudo python3 src/data/to_json.py data/interim/testing.pkl data/interim/test.json 
	$@

create-model:
	sudo python3 src/data/train_data.py data/processed/training.pkl src/docker/xgboost_model.pkl 
	$@

.PHONY: clean
clean:
	sudo rm -f data/raw/*.csv
	sudo rm -f data/raw/*.xls
	sudo rm -f data/processed/*.pkl
	sudo rm -f data/interim/*.pkl
	sudo rm -f data/processed/*.json
	sudo rm -f src/docker/*.pkl

.PHONY: post-test-data
post-test-data: 
	sudo curl -i -X POST -H "Content-Type: application/json" -d @data/interim/test.json http://0.0.0.0:80/json-get-results $@

