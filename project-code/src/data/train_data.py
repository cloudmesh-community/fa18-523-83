import pickle
from xgboost import XGBClassifier
from preprocessing import read_pickle
import click
import pandas as pd


def split_data(df):
    x = df.iloc[:,2:-1]
    y = df.iloc[:,1]
    return x, y


def get_balanced_data(df):
    df_majority = df[df['y']==0]
    df_minority = df[df['y']==1]
    
    df_majority_downsampled = df_majority.sample( n=10026, random_state=123)
    df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    return df_downsampled


def save_model(model, x, y, output_pickle):
    model.fit(x,y)
    with open(output_pickle,'wb') as f:
        pickle.dump(model,f)

@click.command()
@click.argument('input_pickle', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_pickle', type=click.Path(writable=True, dir_okay=False))
def main(input_pickle, output_pickle):
    df=read_pickle(input_pickle)
    df=get_balanced_data(df)
    x_balanced,y_balanced=split_data(df)
    model = XGBClassifier()
    save_model(model, x_balanced, y_balanced, output_pickle)

if __name__ == '__main__':
    main()
