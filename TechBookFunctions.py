"""
This file contains wrappers and helper functions for the technical notebook
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import data_clean as dc


def import_and_clean():
    """This function puts together a few of the cleaning functions from data_clean.py"""
    # data_import.py performs our test-train split and writes four csvs, which we read in here
    X_train_temp = pd.read_csv('./data/dirty_X_train.csv', index_col=0)
    y_train_temp = pd.read_csv('./data/dirty_y_train.csv', index_col=0)
    X_test_temp = pd.read_csv('./data/dirty_X_test.csv', index_col=0)
    y_test_temp = pd.read_csv('./data/dirty_y_test.csv', index_col=0)

    # These perfrom basic data cleaning
    X_train, y_train = dc.data_clean(X_train_temp, y_train_temp)
    X_test, y_test = dc.data_clean(X_test_temp, y_test_temp)

    # these create full sets of test and train data with FIPS county codes for plots
    X_train, X_test, full_data = dc.create_fips_df(X_train, X_test)
    y_train, y_test, full_target = dc.create_fips_df(y_train, y_test)

    return X_train, X_test, y_train, y_test, full_data, full_target


def model_selection_results(search):
    """ This is a wrapper function that turns the results of the crossvalidation into a nice
    dataframe with only relevant info """
    cross_val = pd.DataFrame(search.cv_results_)
    cross_val2 = cross_val.sort_values('rank_test_score')

    nice_table = cross_val2[['rank_test_score', 'param_elastic__alpha',
                             'param_elastic__l1_ratio', 'mean_test_score', 'mean_train_score']]
    return nice_table.head()


def run_model(X_train, y_train, l1_ratio=.5, alpha=.5, save=None):
    """This function takes in the optimal parameters we discovered from crossvalidation, fits a
    elastic net with those parameters to our model, and then displays a barplot of the regression
    coefficients. save is given any truthy value then also save a png of the figure"""
    reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000)
    reg.fit(X_train, y_train)
    coeffs = reg.coef_.tolist()
    coeffs_columns = X_train.columns
    coefficients = list(zip(coeffs, coeffs_columns))
    coefficients.sort()

    plt.figure(figsize=(15, 10))
    plt.xticks(rotation=60)
    plt.title('Effect of Features on Average Life Expectancy')
    plt.ylabel('Expected Change in ALE by Feature ')
    df2 = pd.DataFrame([coeffs_columns, coeffs])
    df_sorted = df2.T
    df_sorted = df_sorted.sort_values(1)
    sns.barplot(list(df_sorted[0]), list(df_sorted[1]), color='#0485d1')
    plt.gca().invert_yaxis()
    if save:
        plt.savefig('Regression_Results.png')

    return df_sorted


def test_model(X_test, y_test, l1_ratio=.5, alpha=.5, X_train=None, y_train=None):
    """This Trains a model on the training data with the specified parameters and then returns
    the score for that model with the test data"""
    reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=50000)
    reg.fit(X_train, y_train)
    print(reg.score(X_test, y_test))

    return reg


def choropleth(full_data, column, title, reverse=None, save=None):
    """This function takes in a data frame and a target column and makes a choropleth. We assume
    that the data frame has a FIPS column if r is any truthy value the color scheme is reveresed,
    and if save is a truthy value then the figure is saved as a png"""
    mind = full_data[column].min()
    maxd = full_data[column].max()

    fips = list(full_data.FIPS)
    values = list(full_data[column])

    bins = list(np.linspace(mind, maxd, 21))
    scale = ["#E50059", "#DA025D", "#D00462", "#C50766", "#BB096B", "#B00B70", "#A60E74",
             "#9C1079", "#91127E", "#871582", "#7C1787", "#721A8C", "#681C90", "#5D1E95",
             "#532199", "#48239E", "#3E25A3", "#3428A7", "#292AAC", "#1F2CB1", "#142FB5",
             "#0A31BA", "#0034BF"]
    if reverse:
        scale.reverse()

    fig = ff.create_choropleth(
        fips=fips, values=values, binning_endpoints=bins, legend_title=title, colorscale=scale)
    fig.layout.template = None

    if save:
        fig.write_image(f'{column}_counties.png')

    fig.write_image("County_LBW.png")
    fig.show()
