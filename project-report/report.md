# Credit Scoring Algorithm and Its Implementation in Production Environment :hand: fa18-523-83


| Nhi Tran
| nytran@iu.edu
| Indiana University
| hid: fa18-523-83
| github: [:cloud:](https://github.com/cloudmesh-community/fa18-523-83/blob/master/project-report/report.md)
| code: [:cloud:](https://github.com/cloudmesh-community/fa18-523-83/blob/master/project-code/)


## Abstract


A credit scoring algorithm is essential to help any bank determine whether to authorize a loan to consumers. Most of the decisions require fast results and high accuracy in order to improve the bank customer satisfaction and profit. Choosing a correct machine learning algorithm will result in high accuracy in the prediction and a reasonable implementation of the algorithm will result in the fast service a bank can provide to its customers.


## Keywords


fa18-523-83, machine learning, predict algorithm, classification, devops, optimization, API

 
## Introduction


For every machine learning problem, there are normally two main areas that everyone focuses on: which machine learning algorithms to use and how to implement and integrate the machine learning code into existing production infrastructure.

Most of the time, machines make predictions by learning and observing the data patterns from previously existing data with known results. Once the training is over, the machine learning code can then be applied to new data and predict the unknown results by applying the trained patterns and algorithms.

The next thing to do after prediction would be determining how to retrieve and apply the result of the prediction into a new or existing production application and how to ensure the continuous deployment into the production environment without resulting deployment code defects.

The business problem is how to determine in real-time whether or not a customer will be experiencing financial distress in the next two years. By predicting the business problem, banking companies can use the results as part of their business rules to decide whether to approve their products to the customer.  

## Design

 

### Dataset


We are utilizing an existing dataset from the 'Give Me Some Credit' competition on Kaggle to train and test algorithms and determine which algorithms would be the best to predict the probability of someone experiencing financial distress in the next two years [@fa18-523-83-credit-dataset].

The Kaggle competition contains a training set, a test set, and a data dictionary. The training set contains 150,000 records of previous customer data with an existing label indicating whether or not each customer had serious bank delinquency within two years. The test set contains about 100,000 records without any label data, which will not be part of the analysis but will be used as part of the benchmarking report.
 
Data descriptions [@fa18-523-83-www-gmsc-kaggle-data] :
*  *SeriousDlqin2yrs*: label data, contains 'Yes' or 'No' indicator 
*  *evolvingUtilizationOfUnsecuredLines*: total balance of unsecured lines such as credit cards and personal lines
*  *age*: bank customers' age
*  *NumberOfTime30-59DaysPastDueNotWorse*: number of times each customer has been 30-59 days past due but no worse in the last 2 years
*  *DebtRatio*: monthly debt payments divided by monthly gross income
*  *MonthlyIncome*: bank customers' monthly income
*  *NumberOfOpenCreditLinesAndLoans*: number of open loans and lines of credit 
*  *NumberOfTimes90DaysLate*: number of times each customer has been 90 days or more past due
*  *NumberRealEstateLoansOrLines*: number of mortgage and real estate loans
*  *NumberOfTime60-89DaysPastDueNotWorse*: number of times each customer has been 60-89 days past due but no worse in the last 2 years
*  *NumberOfDependents*: number of dependents in family excluding themselves (spouse, children etc.)

### Data Visualization


### Data Cleaning

 

### Algorithm Used

 
The goal is to determine whether someone will experience financial distress in the next two years, therefore, there will only be valuable in the label: Yes or No. With a two-labels problem, it is best to use classification algorithms such as Random Forest, XGBoost, LightGBM, Support Vector Machine, Logistic regression.


* **Random Forest**:
* **XGBoost**:
* **LightGBM**:
* **Support Vector Machine**:
* **Logistic Regression**:


### Result Comparison

 


## Implementation

 

### Technologies Used

* Flask API - to allow the ability to pass attributes into production application and receive result in json
* Docker - container
* AWS EC2 - cloud server that will host all the code
* Python - used to built algorithms
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
