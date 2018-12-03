# Credit Scoring Algorithm and Its Implementation in Production Environment fa18-523-83

The business problem that will be focused on is how to determine in real-time 
whether or not a customer will be experiencing financial distress in the next 
two years. By predicting the business problem, banking companies can use the 
results as part of their business rules to decide whether to approve their 
products to the customers.

This project emphasizes the process to determine which machine algorithms 
to use, the steps that are needed to train and test data, and the simple 
implementation of the machine model into a new environment such as AWS 
EC2 Ubuntu Server using Python, Docker and Flask API.


## Getting Started

### Prerequisites

In order the run the code and reproduce the run, the following prerequisites 
need to be met:

* **Ubuntu 18.04 Bionic Beaver or 18.10 Cosmic Cuttlefish Server**: all codes 
were tested on Ubuntu 18.04 and 18.10 Server

* **Kaggle Account**: a Kaggle account is required to pull data from Kaggle.

* **Kaggle API Credentials File**: after the Kaggle account is created, go to 
the *Account* tab of user profile and select *Create API Token* to generate 
and download `kaggle.json` and save as `kaggle.json` at `~/.kaggle` directory 
and set the permission to 600  [@fa18-523-83-www-kaggle-api-github]. 

* **Make**: ensure that *make* is installed. If not, use the following 
command to install *make*:

    ```
	sudo apt-get install make
	```
    
    This will allow the *make* command from *Makefile* to be run. The rest
	of the prerequisites packages and software can be run using *make* command.

* **Project's Git Command**: install git command by running the following 
command:

    ```
	sudo apt-get install git-core
	```

* **Project's Github Repository Cloned**: ensure all project is cloned from 
GitHub using the following command:

    ```sudo git clone https://github.com/cloudmesh-community/fa18-523-83.git```

* **AWS Account** : an AWS account is required to be able to launch a cloud 
server instance for deployment and benchmarking results. The AWS account can 
be created via the AWS EC2 page [@fa18-523-83-www-aws-ec2]. Credit card 
information is required during registration but the server instance can be 
launched for free.

* **Setting up EC2 Server Instance**: once the AWS is created, AWS EC2 
instances can be created. Once logged in, the user can *Launce Instance* 
with the following configurations [@fa18-523-83-www-aws-ec2]:

     *  Step 1: Choose an Amazon Machine Image (AMI) Cancel and Exit: 
	 select *Ubuntu Server 18.04 LTS (HVM), SSD Volume Type*
     *  Step 2: Choose an Instance Type: default to free tier option
     *  Step 3: Configure Instance Details: use all default options 
     *  Step 4: Add Storage: use all default options 
     *  Step 5: Add Tags: use all default options 
     *  Step 6: Configure Security Group: allow SSH (Port 22) and HTTPS 
	 (Port 443) with Source *0.0.0.0/0* and *::/0*
     *  Step 7: Review Instance Launch: after hitting launce, make sure to 
	 choose an existing or create a new key pair and download the key. This 
	 will be the only time user can download a key pair. Ensure the key 
	 permission is 400. 
 
To connect to AWS EC2 server, run the following command:

```
ssh -i <key-file-and-directory> ubuntu@<AWS-public-DNS>
```     

### Installing

#### I. Environment and Files Preparation

After all the prerequisites are met and the Ubuntu server is up and running, 
the follow steps can be used to reproduce the environment and files preparation 
process starting at the directory that the *Makefile* is in:

Step 1: Environment preparation

```
make prepare-environment
```

Step 2: Download data files from Kaggle

```
make download-file
```

Step 3: Prepare Train and Test files:

```
make prepare-files
```

or run the one-step *make* command:

```
make prep-all
```

#### II. Analysis

```
make evaluation
```

#### III. Deployment and testing

Step 1: Build a docker image

```
make docker-build-image
```

Step 2: Run docker image

```
make docker-run-image
```

Step 3: POST JSON test data to API Flask app

In a new terminal, run the following command:

```
make post-test-data
```

The result JSON file should be located in `data/processed/result.json`.


#### IV. Clean Up

The following command to clean up files after the deployment:

```
make clean
```


## Authors

* **Nhi Tran** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

