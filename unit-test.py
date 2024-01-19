import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 03:18:09 2023

@author: ezenwereemmanuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  #
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import matplotlib
import seaborn as sns
from pprint import pprint as pp
from datetime import date
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os  # accessing directory structure
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import random


def years_from_today(date_data):
    """


    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # today_data = date.today()
    date_today = str(date.today())
    date_today = date_today.split("-")
    try:
        date_value = date_data[:]
        date_value = date_value.split("-")
        time_diff = int(date_today[0]) - int(date_value[0]), int(date_today[1]) - int(date_value[1]), int(
            date_today[2]) - int(date_value[2])
        time_years_diff = time_diff[0] + time_diff[1] / 12 + time_diff[2] / 365
        output = time_years_diff

    except AttributeError:
        date_df = date_data.copy()
        for index in range(date_data.shape[0]):
            date_value = date_data[index]
            date_value = date_value.split("-")
            time_diff = int(date_today[0]) - int(date_value[0]), int(date_today[1]) - int(date_value[1]), int(
                date_today[2]) - int(date_value[2])
            time_years_diff = time_diff[0] + time_diff[1] / 12 + time_diff[2] / 365
            date_df[index] = time_years_diff

    output = date_df.astype(int)

    return output


# Sample input data

def age_str_to_num(data):
    """

    Parameters
    ----------
    data : pandas data frame containing eg. 18-24 years old

    Returns
    -------

    """
    data.split(" ")


def print_bar(bar_length=54):
    """
    Display a full bar under a key sentences.
    """
    bar = "-" * bar_length
    print(bar)
    return None


def data_stats(data_frame):
    """

    Parameters
    ----------
    data_frame : pandas data frame.

    Returns
    -------

    """
    print("\nData Stats : ")
    print_bar()
    n_row, n_col = data_frame.shape
    print(f'Data rows - ({n_row}), and Data columns - ({n_col}).\n')
    print('Available Data Types :')
    print_bar()
    print(data_frame.dtypes.value_counts())

    # Classifying Columns into Categorical & Quantitative Variables.
    data_class_summary = {}
    data_class = {}
    # Categorical Variables.
    cat_data = data_frame.select_dtypes(include="object").copy()
    cat_cols = cat_data.columns
    data_class["categorical"] = cat_cols.shape[0]
    data_class_summary[f"categorical ({cat_cols.shape[0]})"] = list(cat_cols)
    # Numerical / Quantitative,  Variables.
    num_data = data_frame.select_dtypes(include='number').copy()
    num_cols = num_data.columns
    data_class_summary[f"quantifiable ({num_cols.shape[0]})"] = list(num_cols)
    data_class["quantifiable"] = num_cols.shape[0]
    print("\nData Class Summary : ")
    print_bar()
    # pp(data_class_summary)
    pp(num_cols)
    pp(list(data_class_summary.keys()))

    # Find the set of columns in the data with missing values.
    df_mean = data_frame.isnull().mean()
    null_cols = set(data_frame.columns[df_mean > 0])
    print(f"\nColumns with missing values ({len(null_cols)}) :")
    print_bar()
    # if not null_cols:
    #     print("None")
    # else:
    #     pp(null_cols)

    # Evaluate the number of missing values in each column.
    null_cols_stats = df_mean * n_row  # df.isnull().sum(axis=0)
    notnull_cols = null_cols_stats[null_cols_stats > 0]
    print('\nNumber of missing values in each Column: ')
    print_bar()
    if notnull_cols.empty:
        print(null_cols_stats)
    else:
        print(notnull_cols)
    print('\nNumber of missing values in each Column as a Percentage (%):')
    print_bar()
    missing_values_ratio = (df_mean[df_mean > 0] * 100).round(2)
    if missing_values_ratio.empty:
        print("None")
    else:
        print(missing_values_ratio)

    return missing_values_ratio


def data_prep(data_frame):
    """
    Parameters
    ----------
    data_frame : pandas data frame
         raw pandas data frame, un prepared for modelling

    Returns
    -------
    prepared_df, a new data frame obtained after removing missing rows,
    displaying quatifiable and categorical columns, and general statistics-
    -data about the data frame.
    """
    # initialize our return df variable.
    prepared_df = data_frame

    # Data Imputation
    min_yrs = 0
    max_yrs = 100
    prepared_df["YearsCodePro"].replace("More than 50 years", max_yrs, inplace=True)
    prepared_df['YearsCode'].replace("More than 50 years", max_yrs, inplace=True)
    prepared_df["YearsCodePro"].replace("Less than 1 year", min_yrs, inplace=True)
    prepared_df['YearsCode'].replace("Less than 1 year", min_yrs, inplace=True)
    prepared_df["YearsCodePro"] = prepared_df["YearsCodePro"].astype('float64')
    prepared_df['YearsCode'] = prepared_df["YearsCodePro"].astype('float64')

    print(prepared_df["YearsCodePro"].dtype)
    # data_stats(prepared_df)
    # Fill numeric columns with the mean

    # num_vars = df.select_dtypes(include=['float', 'int']).columns
    # for col in num_vars:
    #     df[col].fillna((df[col].mean()), inplace=True)
    # data_stats(prepared_df)
    # cat_data = df.select_dtypes(include="object").copy()
    # cat_cols = cat_data.columns

    # Get data stats.
    missing_values_ratio = data_stats(prepared_df)

    # Remove all rows of columns with NAN values.
    threshold = 60
    print(f'\nData-Prep: Drop columns with {threshold}% of their values as NAN')
    print(f'\nShape, pre data-prep: {prepared_df.shape}')
    for col in missing_values_ratio.index:
        if missing_values_ratio[col] > threshold:
            # Drop columns which contain mostly NAN values.
            print(f'missing col {col}-{missing_values_ratio[col]}')
            prepared_df = prepared_df.drop(col, axis=1)
    print(f'Shape, post data-prep: {prepared_df.shape}')
    # missing_values_ratio = data_stats(prepared_df)
    print(f'\nData-Prep: Drop rows containing NAN values')
    prepared_df = prepared_df.dropna(axis=0)
    # print(f'Shape, post second call to data-prep: {prepared_df.shape}')

    # Replacing Categorical Variables with Dummy Variables.
    print(f'\nData-Prep: Replace Categorical Variables with Dummy Variables')
    cat_vars = prepared_df.select_dtypes(include=['object']).copy()
    cat_cols = cat_vars.columns
    for var in cat_cols:
        dummy_var = pd.get_dummies(prepared_df[var], prefix=var, prefix_sep='_', drop_first=True)
        prepared_df = pd.concat([prepared_df.drop(var, axis=1), dummy_var], axis=1)
        # prepared_df = pd.concat([prepared_df.drop(var, axis=1),
        #                          pd.get_dummies(prepared_df[var], prefix=var, prefix_sep='_', drop_first=True)],
        #                          axis=1)
    data_stats(prepared_df)

    return prepared_df


def data_plot(data_frame, independent_var, dependent_var):
    """


    Parameters
    ----------
    data_frame : pandas data frame
         data frame we are intereted in sourcing data to plot.
    response_var :list of strings of pandas data frame colums
         names of cols in the input df we are interested in plotting as response var
    dependent_var : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # plt.figure(figsize=(5, 2.7), layout='constrained')
    # label = str(independent_var[0])+str("-")+str(dependent_var[0])    col_x = independent_var[0]

    # for index_x in range(0, len(independent_var)):
    #     for index_y in range(0, len(dependent_var)):
    #         col_x = independent_var[index_x]
    #         col_y = dependent_var[index_y]
    #         data_frame = data_frame.sort_values(by=[col_x])
    #         x_values = list(data_frame[col_x])[:]  # list(data_frame[col_x][:])
    #         y_values = list(data_frame[col_y])[:]

    #         print(f'\nx axis : {col_x} : {data_frame[col_x].shape}')
    #         print(f"y axis : {col_y} : {data_frame[col_y].shape}")
    #         plt.xlabel(f'{col_x}')
    #         plt.ylabel(f'{col_y}')
    #         plt.title(f'{col_x}-{col_y}')

    #         plt.plot(x_values, y_values, 'o-')

    #         plt.show()

    num_x = random.randint(0, len(independent_var) - 1)
    num_y = random.randint(0, len(dependent_var) - 1)
    col_x = independent_var[num_x]
    col_y = dependent_var[num_y]

    data_frame = data_frame.sort_values(by=[col_x])
    x_values = list(data_frame[col_x])[:]  # list(data_frame[col_x][:])
    y_values = list(data_frame[col_y])[:]

    # Add regression line to plot
    best_fit_params = np.polyfit(x_values, y_values, 1)
    best_fit_poly = np.poly1d(best_fit_params)

    print(f'\nx axis : {col_x} : {data_frame[col_x].shape}')
    print(f"y axis : {col_y} : {data_frame[col_y].shape}")
    plt.xlabel(f'{col_x}')
    plt.ylabel(f'{col_y}')
    plt.title(f'{col_x}-{col_y}')

    plt.plot(x_values, y_values, '-o')
    plt.plot(x_values, best_fit_poly(x_values), color='red')

    plt.show()

    # plt.style.use('_mpl-gallery')

    # # make data
    # num_x = random.randint(0, len(independent_var)-1)
    # num_y1 = random.randint(0, len(dependent_var)-1)
    # num_y2 = random.randint(0, len(dependent_var)-1)
    # col_x = independent_var[num_x]
    # col_y1 = dependent_var[num_y1]
    # col_y2 = dependent_var[num_y2]

    # # np.random.seed(1)
    # # x = np.linspace(0, 8, 16)
    # # y1 = 3 + 4*x/8 + np.random.uniform(0.0, 0.5, len(x))
    # # y2 = 1 + 2*x/8 + np.random.uniform(0.0, 0.5, len(x))

    # x = list(data_frame[col_x])[:1500]  # list(data_frame[col_x][:])
    # y1 = list(data_frame[col_y1])[:1500]
    # y2 = list(data_frame[col_y2])[:1500]

    #     ## make data
    # # x = np.arange(0, 10, 2)
    # # ay = [1, 1.25, 2, 2.75, 3]
    # # by = [1, 1, 1, 1, 1]
    # # cy = [2, 1, 2, 1, 2]
    # y = np.vstack([y1, y2])

    # # plot
    # fig, ax = plt.subplots()

    # ax.stackplot(x, y)

    # ax.set(xlim=(0, 180), xticks=np.arange(1, 180),
    #        ylim=(0, 180), yticks=np.arange(1, 180))

    # plt.show()

    # axs[0].grid(True)
    # plotPerColumnDistribution(prepared_data)
    # plotCorrelationMatrix(prepared_data)
    # plotScatterMatrix(prepared_data)
    # print(prepared_data)


def data_model(data_frame, independent_var, dependent_var):
    """

    Parameters
    ----------
    clean_data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # model fitting.
    # #Drop respondent and expected salary columns
    # df = df.drop(['Respondent', 'ExpectedSalary', 'Salary'], axis=1)

    # #Only use quant variables and drop any rows with missing values
    # num_vars = df[['Salary', 'CareerSatisfaction', 'HoursPerWeek', 'JobSatisfaction', 'StackOverflowSatisfaction']]

    # #Drop the rows with missing salaries
    # drop_sal_df = num_vars.dropna(subset=['Salary'], axis=0)

    # # Mean function
    # fill_mean = lambda col: col.fillna(col.mean())
    # # Fill the mean
    # fill_df = drop_sal_df.apply(fill_mean, axis=0)

    # Split into explanatory and response variables
    X = data_frame[independent_var]
    y = data_frame[dependent_var[1]]

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.10, random_state=42)

    lm_model = LinearRegression()  # Instantiate
    lm_model.fit(X_train, y_train)  # Fit

    # Predict and score the model
    y_test_preds = lm_model.predict(X_test)
    print(f'X test shape : {X_test.shape}, y test shape : {y_test_preds.shape}')
    print(X_test[:20])
    print(y_test_preds[:20].astype('int64'))
    print(y_test[:20].astype('int64'))
    print("\nThe r-squared score for the model using only quant variables was {} on {} values.".format(
        r2_score(y_test, y_test_preds), len(y_test)))

    return lm_model


# Let's start by applying the CRISP-DM PROCESS.
# - Developing Business Understanding.
# - Developing Data Understanding.
# - Data Preparation.
# - Modelling the Data
# - Evaluating the Model
df = pd.read_csv('StackOverflow-data/data-dir/stack-overflow-developer-survey-2023/survey_results_public.csv')
# print(df.head(5))
# df = pd.read_csv(dir)

# CRISP DM - Data Understanding
response_var = ['ResponseId',
                'Q120',
                'MainBranch',
                'Age',
                'Employment',
                'RemoteWork',
                'CodingActivities',
                'EdLevel',
                'LearnCode',
                'LearnCodeOnline',
                'LearnCodeCoursesCert',
                'YearsCode',
                'YearsCodePro',
                'DevType',
                'OrgSize',
                'PurchaseInfluence',
                'TechList',
                'BuyNewTool',
                'Country',
                'Currency',
                'CompTotal',
                'LanguageHaveWorkedWith',
                'LanguageWantToWorkWith',
                'DatabaseHaveWorkedWith',
                'DatabaseWantToWorkWith',
                'PlatformHaveWorkedWith',
                'PlatformWantToWorkWith',
                'WebframeHaveWorkedWith',
                'WebframeWantToWorkWith',
                'MiscTechHaveWorkedWith',
                'MiscTechWantToWorkWith',
                'ToolsTechHaveWorkedWith',
                'ToolsTechWantToWorkWith',
                'NEWCollabToolsHaveWorkedWith',
                'NEWCollabToolsWantToWorkWith',
                'OpSysPersonal use',
                'OpSysProfessional use',
                'OfficeStackAsyncHaveWorkedWith',
                'OfficeStackAsyncWantToWorkWith',
                'OfficeStackSyncHaveWorkedWith',
                'OfficeStackSyncWantToWorkWith',
                'AISearchHaveWorkedWith',
                'AISearchWantToWorkWith',
                'AIDevHaveWorkedWith',
                'AIDevWantToWorkWith',
                'NEWSOSites',
                'SOVisitFreq',
                'SOAccount',
                'SOPartFreq',
                'SOComm',
                'SOAI',
                'AISelect',
                'AISent',
                'AIAcc',
                'AIBen',
                'AIToolInterested in Using',
                'AIToolCurrently Using',
                'AIToolNot interested in Using',
                'AINextVery different',
                'AINextNeither different nor similar',
                'AINextSomewhat similar',
                'AINextVery similar',
                'AINextSomewhat different',
                'TBranch',
                'ICorPM',
                'WorkExp',
                'Knowledge_1',
                'Knowledge_2',
                'Knowledge_3',
                'Knowledge_4',
                'Knowledge_5',
                'Knowledge_6',
                'Knowledge_7',
                'Knowledge_8',
                'Frequency_1',
                'Frequency_2',
                'Frequency_3',
                'TimeSearching',
                'TimeAnswering',
                'ProfessionalTech',
                'Industry',
                'SurveyLength',
                'SurveyEase',
                'ConvertedCompYearly']

# - Data Preparation
print("\nBeginning Data Preparation Steps ... : \n")
prepared_data = data_prep(df)
independent_var = ['YearsCode', 'YearsCodePro', 'WorkExp']
dependent_var = ['CompTotal', 'ConvertedCompYearly']

# - Data Visualization
data_plot(prepared_data, independent_var, dependent_var)

# - Model.
data_model(prepared_data, independent_var, dependent_var)

# - Evaluation
# dataModel.fit(x,y)
# data_plot(x,y)


