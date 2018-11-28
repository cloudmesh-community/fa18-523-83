# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 20:26:39 2018

@author: NhiYobie
"""


import pandas as pd
import click

def read_raw_data(input_file):
    df = pd.read_csv(input_file, header=None,sep=',')
    return df



def preprocessing(training):
    
    #Replace missing or bad data with median value
    training['NumberOfDependents']= training['NumberOfDependents'].fillna((training['NumberOfDependents'].median()))
    training['MonthlyIncome']= training['MonthlyIncome'].fillna(training['MonthlyIncome'].median())
    training_prep=training.replace({'age': {0: training['age'].median()}}) 
    
    #drop DebtRatio due to low correllation 
    training_prep=training_prep.drop('DebtRatio', 1)
    
    #feature engineering by adding highly correlated features into one new feature

    training_prep['TotalNumberofTimesPastDue']= training_prep['NumberOfTime30-59DaysPastDueNotWorse'] + training_prep['NumberOfTime60-89DaysPastDueNotWorse'] + training_prep['NumberOfTimes90DaysLate'] 
    training_prep=training_prep.drop('NumberOfTime30-59DaysPastDueNotWorse', 1)
    training_prep=training_prep.drop('NumberOfTime60-89DaysPastDueNotWorse', 1)
    training_prep=training_prep.drop('NumberOfTimes90DaysLate', 1)
    
    #drop highly correlated features 
    training_prep['TotalNumberOfOpenLines']= training_prep['NumberOfOpenCreditLinesAndLoans'] + training_prep['NumberRealEstateLoansOrLines'] 
    training_prep=training_prep.drop('NumberOfOpenCreditLinesAndLoans', 1)
    training_prep=training_prep.drop('NumberRealEstateLoansOrLines', 1)

    #drop TotalNumberOfOpenLines due to low correllation 
    training_prep=training_prep.drop('TotalNumberOfOpenLines', 1)
    
    return training_prep

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_pickle', type=click.Path(writable=True, dir_okay=False))

def main(input_file, output_pickle):
    print('Preprocessing data file: ', input_file)
    df = read_raw_data(input_file)
    df = preprocessing(df)
    df.to_pickle(output_pickle)
    
    
if __name__ == '__main__':
    main()
