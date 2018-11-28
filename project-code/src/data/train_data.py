import click
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from preprocessing import read_pickle


def simple_label(df):
    df = ['id','y','x1','x2','x3','x4','x5','x6','x7','x8','x9']
    return df

def get_balanced_data(df):
    df_majority = df[df['y']==0]
    df_minority = df[df['y']==1]
    
    df_majority_downsampled = df_majority.sample( n=10026, random_state=123)
    df_downsampled = pandas.concat([df_majority_downsampled, df_minority])
	
	
def split_data(df):
    x = df.iloc[:,2:-1]
    y = df.iloc[:,1]
    return x, y



def LogisticRegression():
    algorithm= LogisticRegression(solver='liblinear')
    return algorithm


def LogisticRegression_Penalized(x_train,y_train, x_test, y_test):
    algorithm= LogisticRegression(C=500,penalty='l1', tol=0.10, solver='saga')
    return algorithm


def RandomForestClassifier()
    algorithm= RandomForestClassifier()
    return algorithm



def model_training(x_train,y_train, algorithm)
    model=algorithm.fit(x_train, y_train)
    return model

def model_prediction(x_test, model)
    y_pred = model.predict(x_test)
    return y_pred

def main():
	
    df=read_pickle(data/processed/training.pkl)
    df=simple_label(df)
    x,y=split_data(df)	
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
	








