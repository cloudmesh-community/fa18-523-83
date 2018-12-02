# -*- coding: utf-8 -*-

import pandas as pd
import click

def read_raw_data(input_file):
    df = pd.read_csv(input_file, sep=',')
    return df

def read_pickle(input_file):
    df = pd.read_pickle(input_file)
    return df

def preprocessing(df):
    
    #Replace missing or bad data with median value
    df['NumberOfDependents']= df['NumberOfDependents'].fillna((df['NumberOfDependents'].median()))
    df['MonthlyIncome']= df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df_prep=df.replace({'age': {0: df['age'].median()}}) 
    
    #drop DebtRatio due to low correllation 
    df_prep=df_prep.drop('DebtRatio', 1)
    
    #feature engineering by adding highly correlated features into one new feature

    df_prep['TotalNumberofTimesPastDue']= df_prep['NumberOfTime30-59DaysPastDueNotWorse'] + df_prep['NumberOfTime60-89DaysPastDueNotWorse'] + df_prep['NumberOfTimes90DaysLate'] 
    df_prep=df_prep.drop('NumberOfTime30-59DaysPastDueNotWorse', 1)
    df_prep=df_prep.drop('NumberOfTime60-89DaysPastDueNotWorse', 1)
    df_prep=df_prep.drop('NumberOfTimes90DaysLate', 1)
    
    #drop highly correlated features 
    df_prep['TotalNumberOfOpenLines']= df_prep['NumberOfOpenCreditLinesAndLoans'] + df_prep['NumberRealEstateLoansOrLines'] 
    df_prep=df_prep.drop('NumberOfOpenCreditLinesAndLoans', 1)
    df_prep=df_prep.drop('NumberRealEstateLoansOrLines', 1)

    #drop TotalNumberOfOpenLines due to low correllation 
    df_prep=df_prep.drop('TotalNumberOfOpenLines', 1)



    return df_prep

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
