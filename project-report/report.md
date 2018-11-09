# Credit Scoring Algorithm and Its Implementation in Production Environment :hand: fa18-523-83


| Nhi Tran
| nytran@iu.edu
| Indiana University
| hid: fa18-523-83
| github: [:cloud:](https://github.com/cloudmesh-community/fa18-523-83/blob/master/project-report/report.md)
| code: [:cloud:](https://github.com/cloudmesh-community/fa18-523-83/blob/master/project-code/)


## Abstract


A credit scoring algorithm is essential to help bank determine whether to authorize a loan to consumers. Most of the decisions require fast results and high accuracy in order to improve the bank customer satisfaction and profit. Choosing a correct machine learning algorithm will result in high accuracy in the prediction and a reasonable implementation of the algorithm will result in the fast service a bank can provide to its customers.


## Keywords


fa18-523-83, machine learning, predict algorithm, classification, devops, optimization, api

 
## Introduction


For every machine learning problem, there are normally two main areas that everyone focus on: which machine learning algorithms to use and how to implement and integrate the machine learning code into existing production infrastructure.

Most of the time, machine makes predictions by learning and observing the data patterns from previous existing data with known results. Once the training is over, the machine learning code can then be applied to new data and predict the unknown results by applying the trained patterns.

Once the code is completed, the next thing to do would be determining how to retrieve and apply the result of the prediction into a new or existing production application and how to ensure the continuous deployment into production environment without resulting deployment code defects.

 
## Design

 

### Dataset


We are utilizing an existing dataset on Kaggle to train and test algorithms and determine which algorithms would be the best to predict the probability of someone experiencing financial distress in the next two years [@fa18-523-83-credit-dataset].

Data descriptions:

* RevolvingUtilizationOfUnsecuredLines: Total balance on credit cards and personal lines of credit except real estate and no installment debt like car loans divided by the sum of credit limits - percentage
* age : Borrowers' age
* NumberOfTime30-59DaysPastDueNotWorse: Number of times borrower has been 30-59 days past due but no worse in the last 2 years.
* DebtRatio: Monthly debt payments, alimony,living costs divided by monthy gross income percentage
* MonthlyIncome: Borrowers' monthly income
* NumberOfOpenCreditLinesAndLoans: Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)
* NumberOfTimes90DaysLate: Number of times borrower has been 90 days or more past due.
* NumberRealEstateLoansOrLines: Number of mortgage and real estate loans including home equity lines of credit
* NumberOfTime60-89DaysPastDueNotWorse: Number of times borrower has been 60-89 days past due but no worse in the last 2 years.
* NumberOfDependents: Number of dependents in family excluding themselves (spouse, children etc.)

The Kaggle's dataset contains 150,000 in training population and 102,000 in testing population.


### Data Visualization


### Data Cleaning

 

### Algorithm Used

 
The goal is to determing whether someone will experience financial distress in the next two years, therefore, there will only be value in the label: Yes or No. With a two-labels problem, it is best to use classification algorithms such as Random Forest, XGBoost, LightGBM, Support Vector Machine, Logistic regression.


* **Random Forest**:
* **XGBoost**:
* **LightGBM**:
* **Support Vector Machine**:
* **Logistic Regression**:


### Result Comparison

 


## Implementation

 

### Technologies Used

*             Flask API - to allow the ability to pass attributes into production application and receive result in json
*             Docker - container
*             AWS EC2 - cloud server that will host all the code
*             Python - used to built algorithms
               *  Panda
                                *  Numpy
                                *  Scikit-learn
                                ?   Etc.


## Results


### Deployment Benchmarks


### Application Benchmarks


## Limitations


## Conclusion


## Acknowledgements

 
## References
