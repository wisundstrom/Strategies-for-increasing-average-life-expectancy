"""

"""
import pandas as pd
import numpy as np
from data_clean import data_clean
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.pipeline import Pipeline, make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt


def import_and_clean():
    # data_import.py performs our test-train split and writes four csvs, which we read in here
    X_train_temp = pd.read_csv('./data/dirty_X_train.csv', index_col=0)
    y_train_temp = pd.read_csv('./data/dirty_y_train.csv', index_col=0)
    X_test_temp = pd.read_csv('./data/dirty_X_test.csv', index_col=0)
    y_test_temp = pd.read_csv('./data/dirty_y_test.csv', index_col=0)

    # We then use the function from data_clean.py
    X_train,y_train = data_clean(X_train_temp,y_train_temp)
    X_test,y_test = data_clean(X_test_temp,y_test_temp)
    return X_train, X_test, y_train, y_test


def model_selection_results(search):
    CV = pd.DataFrame(search.cv_results_)
    CV2 = CV.sort_values('rank_test_score')

    nice_table = CV2[['rank_test_score', 'param_elastic__alpha',
                      'param_elastic__l1_ratio', 'mean_test_score', 'mean_train_score']]
    return nice_table.head()


def run_model(X_train, y_train, l1_ratio=.5, alpha=.5, save=None):
    reg = ElasticNet(alpha=.4, l1_ratio=.6, max_iter=50000)
    reg.fit(X_train, y_train)
    coeffs = reg.coef_.tolist()
    coeffs_columns = X_train.columns
    coefficients = list(zip(coeffs, coeffs_columns))
    coefficients.sort()

    plt.figure(figsize=(15,10))
    plt.xticks(rotation=60)
    plt.title('Effect of Features on Average Life Expectancy')
    plt.ylabel('Expected Change in ALE by Feature ')
    df2=pd.DataFrame([coeffs_columns,coeffs])
    df_sorted=df2.T
    df_sorted=df_sorted.sort_values(1)
    sns.barplot(list(df_sorted[0]),list(df_sorted[1]))
    plt.gca().invert_yaxis()
    if save:
        plt.savefig('Regression_Results.png')
    
    return df_sorted
