#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 03:18:09 2023
@author: ezenwereemmanuel
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint as pp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns


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
    try:
        n_row, n_col = data_frame.shape

    except ValueError:
        n_row = 'generic'
        n_col = 'generic'

    num_rows = "{:0,.0f}".format(n_row)
    print(f'Data rows - ({num_rows}), and Data columns - ({n_col}).\n')
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
    # pp(num_cols)
    pp(list(data_class_summary.keys()))

    # Find the set of columns in the data with missing values.
    df_mean = data_frame.isnull().mean()
    null_cols = set(data_frame.columns[df_mean > 0])
    print(f"\nColumns with missing values: ({len(null_cols)})")
    print_bar()

    # Evaluate the number of missing values in each column.
    null_cols_stats = df_mean * n_row
    notnull_cols = null_cols_stats[null_cols_stats > 0]
    print('\nNumber of missing values per Column: ')
    print_bar()
    if notnull_cols.empty:
        print(null_cols_stats)
    else:
        print(notnull_cols)
    print('\nNumber of missing values per Column (%):')
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
    prepared_df = data_frame.copy()

    # Data Imputation
    min_yrs = 0
    max_yrs = 100
    prepared_df["YearsCodePro"].replace(
        "50+", max_yrs, inplace=True)
    prepared_df['YearsCode'].replace(
        "50+", max_yrs, inplace=True)
    prepared_df["YearsCodePro"].replace(
        "<1", min_yrs, inplace=True)
    prepared_df['YearsCode'].replace("Less than 1 year", min_yrs, inplace=True)
    prepared_df["YearsCodePro"] = prepared_df["YearsCodePro"].astype('float64')
    prepared_df['YearsCode'] = prepared_df["YearsCodePro"].astype('float64')

    # Get data stats.
    missing_values_ratio = data_stats(prepared_df)

    # Remove all rows of columns with NAN values.
    threshold = 60
    print(
        f'\nData-Prep: Drop columns with {threshold}% of their values as NAN')
    print(f'\nShape, pre data-prep: {prepared_df.shape}')
    for col in missing_values_ratio.index:
        if missing_values_ratio[col] > threshold:
            prepared_df = prepared_df.drop(col, axis=1)
    print(f'Shape, post data-prep: {prepared_df.shape}')
    print('\nData-Prep: Drop rows containing NAN values')
    prepared_df = prepared_df.dropna(axis=0)
    print(f'Shape, Post removal of NAN column rows: {prepared_df.shape}')

    # Replacing Categorical Variables with Dummy Variables.
    print('\nData-Prep: Replace Categorical Variables with Dummy Variables')
    cat_vars = prepared_df.select_dtypes(include=['object']).copy()
    cat_cols = cat_vars.columns
    for var in cat_cols:
        dummy_var = pd.get_dummies(
            prepared_df[var], prefix=var, prefix_sep='_', drop_first=True)
        prepared_df = pd.concat(
            [prepared_df.drop(var, axis=1), dummy_var], axis=1)

    data_stats(prepared_df)

    return prepared_df


def data_plot(x_label, y_label, x_values, y_values, path):
    """
    

    Parameters
    ----------
    x_label : TYPE
        DESCRIPTION.
    y_label : TYPE
        DESCRIPTION.
    x_values : TYPE
        DESCRIPTION.
    y_values : TYPE
        DESCRIPTION.
    path : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    print("\n\nData Plots")
    print_bar()

    if type(x_values[0]) == list or type(x_values[0]) == pd.core.series.Series:
        colors = ['b', 'g', 'c', 'm', 'y', 'k']
        fig, ax = plt.subplots()
        for index in range(0, len(x_values)):
            x_val = x_values[index]
            y_val = y_values[index]
            LABEL = 'Model Prediction'

            if index == 0:
                # Add regression line to plot
                best_fit_params = np.polyfit(x_val, y_val, 1)
                best_fit_poly = np.poly1d(best_fit_params)
                ax.plot(x_val, best_fit_poly(x_val),
                        color='r', label='Line Of Best Fit')
                LABEL = 'Actual'

            ax.set_xlabel(f'{x_label}')
            ax.set_ylabel(f'{y_label}')
            ax.set_title(f'{x_label} vs {y_label}')

            color = colors[min(index, len(colors))]
            ax.plot(x_val, y_val, 'o-', color=color, label=LABEL)
            ax.legend()


    else:
        # Add regression line to plot
        best_fit_params = np.polyfit(x_values, y_values, 1)
        best_fit_poly = np.poly1d(best_fit_params)

        plt.xlabel(f'{x_label}')
        plt.ylabel(f'{y_label}')
        plt.title(f'{x_label} vs {y_label}')

        plt.plot(x_values, y_values, 'o-', color='b')
        plt.plot(x_values, y_values, 'o-', color='b')
        plt.plot(x_values, best_fit_poly(x_values), color='red')
    plt.savefig(path)
    plt.show()



def data_eval(model, data_frame, output_var):
    """
    Parameters
    ----------
    model : LinearRegression model
    data_frame : pandas data frame.
    output_var : TYPE
        DESCRIPTION.

    Returns
    -------
    y_preds :pandas series, output predictions of our model.
    y_test : pandas series, outputs from our data frame to test our model's performance.
    X_test : pandas data frame containing our input variables.
    r2_val : r2 score to evlauate the performance of our model
    mse_val : mean squre error to evaluate the performance of our model. 
    """
    # model fitting.

    # Split into explanatory and response variables
    y = data_frame[output_var[1]]

    for col in output_var:
        data_frame.drop(col, inplace=True, axis=1)

    X = data_frame
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.10, random_state=42)
    model.fit(X_train, y_train)  # Fit

    # Predict using the Model
    y_preds = model.predict(X_test)

    # Score the model
    r2_val = r2_score(y_test, y_preds)
    mse_val = mean_squared_error(y_test, y_preds)

    return y_preds, y_test, X_test, r2_val, mse_val




# -----------------------------------------------------------------------------
# Data Science Capstone Project 1: 
# -----------------------------------------------------------------------------
# 1) Who are the Top X respondents ?
# 2) What is common amongst the Top X, earning respondent in each age group?.
# 3) What is common amongst the top earning respondent across age groups?.
# 4) What is common amongst the Least X earning respondent.
# 5) What is common amongst the Least X earning respondent across age groups?.
# 6) What is common amongst the young Top earners of age 25-34 Yeras.? 


#------------------------------------------------------------------------------
print_bar()
print("Data Science analysis of the StackOverFlow SURVEY DATA-2023")
print_bar()
df = pd.read_csv('StackOverflow-data/data-dir/stack-overflow-developer-survey-2023/survey_results_public.csv')


# Data Preparation
# -----------------------------------------------------------------------------
data_frame = df.copy()
data_frame['OrgSize'] = data_frame['OrgSize'].str.replace(' employees', '')
data_frame['OrgSize'] = data_frame['OrgSize'].str.replace(' to ', '-')
data_frame['OrgSize'] = data_frame['OrgSize'].str.replace(' or more', '+')
data_frame['OrgSize'] = data_frame['OrgSize'].str.replace(
    'Just me - I am a freelancer, sole proprietor, etc.', '1')
data_frame['OrgSize'] = data_frame['OrgSize'].str.replace('I don’t know', '?')


data_frame['Age'] = data_frame['Age'].str.replace(' years old', '')
data_frame['Age'] = data_frame['Age'].str.replace('Under', '<')
data_frame['Age'] = data_frame['Age'].str.replace(' years or older', '+')

data_frame['RemoteWork'] = data_frame['RemoteWork'].str.replace('Hybrid (some remote, some in-person)', 'Hybrid')

data_frame['Industry'] = data_frame['Industry'].str.replace('Information Services, IT, Software Development, or other Technology', 'IT')
data_frame['Industry'] = data_frame['Industry'].str.replace('Financial Services', 'F')
data_frame['Industry'] = data_frame['Industry'].str.replace('Manufacturing, Transportation, or Supply Chain', 'MTS')
data_frame['Industry'] = data_frame['Industry'].str.replace('Healthcare', 'H')
data_frame['Industry'] = data_frame['Industry'].str.replace('Retail and Consumer Services', 'HC')
data_frame['Industry'] = data_frame['Industry'].str.replace('Legal Services', 'L')
data_frame['Industry'] = data_frame['Industry'].str.replace('Wholesale', 'W')
data_frame['Industry'] = data_frame['Industry'].str.replace('Oil & Gas', 'O')
data_frame['Industry'] = data_frame['Industry'].str.replace('Advertising Services', 'Ads')
data_frame['Industry'] = data_frame['Industry'].str.replace('Insurance', 'I')
data_frame['Industry'] = data_frame['Industry'].str.replace('Higher Education', 'Edu')

data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Bachelor’s degree (B.A., B.S., B.Eng., etc.)', 'B.Sc/B.Eng/B.A')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Master’s degree (M.A., M.S., M.Eng., MBA, etc.)', 'M.')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Some college/university study without earning a degree', 'no_degree')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)', 'Sec')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Professional degree (JD, MD, Ph.D, Ed.D, etc.)', 'Doctorate')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Associate degree (A.A., A.S., etc.)', 'Associate')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Primary/elementary school', 'Pri')
data_frame['EdLevel'] = data_frame['EdLevel'].str.replace('Something else', 'Others')

data_frame['Currency'] = data_frame['Currency'].str.replace('USD\tUnited States dollar', 'US$')
data_frame['Currency'] = data_frame['Currency'].str.replace('GBP\tPound sterling', '£')
data_frame['Currency'] = data_frame['Currency'].str.replace('HKD\tHong Kong dollar', 'HKD')

data_frame['YearsCodePro'] = data_frame['YearsCodePro'].str.replace('More than 50 years', '50+')
data_frame['YearsCodePro'] = data_frame['YearsCodePro'].str.replace('Less than 1 year', '<1')

print(df.head(5))
# df = pd.read_csv(dir)
prepared_data = data_prep(data_frame.copy())


# Data Visualization - Correlation Map
# -----------------------------------------------------------------------------
print("\n\n\n\n\nVisualizing our data...")
save_dir = "Data/"
plt.style.use("seaborn-v0_8")
# - - plot multicollinearity between independent features using a heatmap
input_var = ['YearsCode', 'YearsCodePro', 'WorkExp', "EdLevel", ]
output_var = ['CompTotal', 'ConvertedCompYearly']

color = plt.get_cmap('RdYlGn')
color.set_bad('lightblue')
df_corr = df.corr(numeric_only=True)
sns.heatmap(data=df_corr, annot=True, cmap=color)
plt.savefig(save_dir+"correlation_map.png")


histplot_columns = ["WorkExp"]
bar_chart_columns = ["OrgSize", "Age", "RemoteWork", "Industry", "DevType"]
for column_name in histplot_columns:
    plot_data = data_frame[column_name].dropna(axis=0)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(plot_data, align='mid')
    ax.set_xlabel(f"{column_name}")
    ax.set_ylabel("Frequency")
    title = f"DataVisualiztion-{column_name}"
    ax.set_title(title)
    plt.savefig(save_dir+title+".png")
    plt.show()


for column_name in bar_chart_columns:
    plot_data = data_frame[column_name]
    plot_data = plot_data.value_counts()
    fig, ax = plt.subplots(figsize=(15, 6))
    if column_name == "DevType": 
        ax.barh(plot_data.index, plot_data.values)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_ylabel(f"{column_name}")
    else: 
        plt.bar(plot_data.index, plot_data.values)
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel("Frequency")
    title = f"DataVisualiztion-{column_name}"
    ax.set_title(title)
    plt.savefig(save_dir+title+".png")
    plt.show()



# - Data Modelling.
# -----------------------------------------------------------------------------
# - Who are the Top X earning respondents, Focus on UK & USA. ------- Plot
print("\n\nQ1) Who are the Top X earning respondents, Focus on UK & USA : \n")
print_bar()
print("Top earning Respondents")
print_bar()


L = ['United States of America', 'United Kingdom of Great Britain and Northern Ireland']
column_names = ["ConvertedCompYearly", "Age", "WorkExp", "OrgSize", "RemoteWork", "EdLevel", "LearnCode", "LearnCodeOnline", "LearnCodeCoursesCert", "YearsCode", "YearsCodePro", "Currency", "DevType", "LanguageHaveWorkedWith", "Industry", "Country"]
x = 1000
# create a Boolean mask for the rows 
mask = data_frame['Country'].isin(L)
df_model = data_frame#[mask]
filter_ =  "ConvertedCompYearly"
top_x_ids = df_model[filter_].nlargest(x).index.tolist()
top_earners = data_frame.iloc[top_x_ids]


# -----------------------------------------------------------------------------
category = 'TopEarner-' 
numeric_names = ["ConvertedCompYearly","WorkExp", "OrgSize"]
category = "TopEarners-"
histplot_columns = ["WorkExp"]
bar_chart_columns = ["OrgSize", "Age", "RemoteWork", "Industry", "DevType"]
for column_name in histplot_columns:
    plot_data = top_earners[column_name].dropna(axis=0)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(plot_data, align='mid')
    ax.set_xlabel(f"{column_name}")
    ax.set_ylabel("Frequency")
    title = f"{category}-{column_name}"
    ax.set_title(title)
    plt.savefig(save_dir+title+".png")
    plt.show()


for column_name in bar_chart_columns:
    plot_data = top_earners[column_name]
    plot_data = plot_data.value_counts()
    fig, ax = plt.subplots(figsize=(15, 6))
    if column_name == "DevType": 
        ax.barh(plot_data.index, plot_data.values)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_ylabel(f"{column_name}")
    else: 
        plt.bar(plot_data.index, plot_data.values)
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel("Frequency")
    title = f"{category}-{column_name}"
    ax.set_title(title)
    plt.savefig(save_dir+title+".png")
    plt.show()
# -----------------------------------------------------------------------------



# - What is common amongst the Top X, earning respondent ?------- Code.

# -----------------------------------------------------------------------------
print("\n\nQ2) What is common amongst the Top X, earning respondent : \n")


for column in column_names:
    distr = top_earners[column].value_counts()
    xmost_freq = distr.head(1)#.index.tolist()
    indices = xmost_freq.index
    values = xmost_freq.values
    count = sum(distr.values)
    max_freq = distr.max()
    #print(f"\nTop {format(int(x), ',')} Earners - {column}: ")
    #print_bar()
    print(f'\n{column}:')
    for index in range(len(indices)): 
        print(f'{indices[index]} -------- {round((values[index]/count) * 100, 2)}%')




# Who are the Least X earning respondents ---------- Code.
# -----------------------------------------------------------------------------
print("\n\nQ3) Who are the Least X earning respondents : \n")

print("\n\n")
print_bar()
print("Least earning Respondents")
print_bar()


mask = data_frame['Country'].isin(L)
df_model = data_frame#[mask]
column_name =  "ConvertedCompYearly"
least_x_ids = df_model["ConvertedCompYearly"].nsmallest(x).index.tolist()
least_earners = data_frame.iloc[least_x_ids]


# - What is common amongst the Least X earning respondents  ------------- Code.
# -----------------------------------------------------------------------------
print("\n\nQ4) What is common amongst the Least X earning respondents : \n")

for column in column_names:
    distr = least_earners[column].value_counts()
    xmost_freq = distr.head(2)
    indices = xmost_freq.index
    values = xmost_freq.values
    count = sum(distr.values)
    max_freq = distr.max()
    #print(f"\nLeast {format(int(x), ',')} Earners - {column}: ")
    #print_bar()
    # print(f'\n{column}: {indices[0]} -------- {round((values[0]/count) * 100, 2)}%')
    print(f'\n{column}:')
    # for index in range(len(indices)): 
    #     print(f'{indices[index]} -------- {round((values[index]/count) * 100, 2)}%')
    if column == 'ConvertedCompYearly':
        for index in range(len(indices)): 
            print(f'{indices[index]} -------- {round((values[index]/count) * 100, 2)}%')
    else:
        print(f'{indices[0]} -------- {round((values[0]/count) * 100, 2)}%')

 
# -----------------------------------------------------------------------------
category = 'LeastEarners-' 
numeric_names = ["ConvertedCompYearly","WorkExp", "OrgSize"]
histplot_columns = ["WorkExp"]
bar_chart_columns = ["OrgSize", "Age", "RemoteWork", "Industry", "DevType"]
for column_name in histplot_columns:
    plot_data = least_earners[column_name].dropna(axis=0)
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.hist(plot_data, align='mid')
    ax.set_xlabel(f"{column_name}")
    ax.set_ylabel("Frequency")
    title = f"{category}-{column_name}"
    ax.set_title(title)
    plt.savefig(save_dir+title+".png")
    plt.show()


for column_name in bar_chart_columns:
    plot_data = least_earners[column_name]
    plot_data = plot_data.value_counts()
    fig, ax = plt.subplots(figsize=(15, 6))
    if column_name == "DevType": 
        ax.barh(plot_data.index, plot_data.values)
        ax.invert_yaxis()
        ax.set_xlabel("Frequency")
        ax.set_ylabel(f"{column_name}")
    else: 
        plt.bar(plot_data.index, plot_data.values)
        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel("Frequency")
    title = f"{category}-{column_name}"
    ax.set_title(title)
    plt.savefig(save_dir+title+".png")
    plt.show()

# -----------------------------------------------------------------------------       
# Data Visualization
# -----------------------------------------------------------------------------
print("\n\n")
print_bar()
category = "Comparative Plots -"
print(category)
print_bar()
# - Visualization - (Top / Least)
for column in column_names: #top_earners.columns[1:]:
    hist_data_top = top_earners[column]
    hist_data_least = least_earners[column]
    
    
    if len(list(hist_data_top.value_counts().index)) > 10: 
        # print most common value of top earners
        hist_data_top.head(1)
        # print most common value of least earners
        distr_least = least_earners[column].value_counts()
        distr_top = top_earners[column].value_counts()
        xmost_freq_least = distr_least.head(1)
        xmost_freq_top = distr_top.head(1)
        indices_least = xmost_freq_least.index
        values_least = xmost_freq_least.values
        indices_top = xmost_freq_top.index
        values_top = xmost_freq_top.values
        count = sum(distr.values)
        max_freq_top = distr_top.max()
        max_freq_least = distr_least.max()
        print(f'\n\nSummary :{column}')
        print_bar()
        print(f'Most Top Earners: {indices_top[0]} -------- {round((values_top[0]/count) * 100, 2)}%')
        print(f'Most Least Earners : {indices_least[0]} -------- {round((values_least[0]/count) * 100, 2)}%')
        print_bar()
        print("\n")
    
    else:
        
        
        if column not in numeric_names:
            hist_data_top = top_earners[column].astype(str)
            hist_data_least = least_earners[column].astype(str)
        
        if column == 'ConvertedCompYearly':
            max_income = hist_data_top.value_counts().index.max()
            fig, ax = plt.subplots()
            plt.hist([hist_data_top, hist_data_least], bins=25, label=['top', 'least'])
            tick_labels = np.arange(0,max_income, (max_income/8 + 1000 - 1) // 1000 * 1000)
            plt.xticks(tick_labels)
            ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            #plt.savefig(save_dir+category+column+".png")
            #plt.show()
        
        elif column == 'OrgSize':
            fig, ax = plt.subplots()
            hist_data_top =  hist_data_top.astype(str)
            hist_data_least = hist_data_least.astype(str)
            tick_labels = list(hist_data_top.value_counts().index)
            bins_val = np.arange(len(tick_labels)+1) - 0.5
            plt.hist([hist_data_top, hist_data_least], bins=bins_val, label=['top', 'least'])
            #plt.savefig(save_dir+category+column+".png")
            #plt.show()
                
        else:  
            bins = np.linspace(-10, 10, 30)
            #plt.hist(hist_data_top, label=['top', 'least'])
            plt.hist([hist_data_top, hist_data_least], label=['top', 'least'])
            #plt.savefig(save_dir+category+column+".png")
           # plt.show()
            
        
        plt.legend(loc='upper right')
        plt.xlabel(f"{column}")
        plt.ylabel("Frequency")
        title = category+column
        plt.title(title)
        plt.savefig(save_dir+title+".png")
        plt.show()
        
        # print most common value of top earners
        hist_data_top.head(1)
        # print most common value of least earners
        distr_top = top_earners[column].value_counts()
        xmost_freq_top = distr_top.head(1)
        indices_top = xmost_freq_top.index
        values_top = xmost_freq_top.values
        count = sum(distr.values)
        max_freq_top = distr_top.max()
        print(f'\nSummary : {column}')
        print_bar()
        print(f'Most Top Earners: {indices_top[0]} -------- {round((values_top[0]/count) * 100, 2)}%')
        print("\n")
        


# What is the major contributor to the difference in income between the -
# - Top Earners and Least Earners ? 
# -----------------------------------------------------------------------------

print("\nQ5) What is the major contributor to difference in income between the Top Earners and Least Earners : \n")
# print_bar()
record = {}
for column in top_earners.columns[1:]: 
    distr_top = top_earners[column].value_counts()
    distr_least = least_earners[column].value_counts()
    most_freq_top = distr_top.head(1)#.index.tolist()
    most_freq_least = distr_least.head(1)
    
    
    value_top = most_freq_top.index[0]
    freq_top = most_freq_top.values[0]
    freq_least = most_freq_least.values[0]
    value_least = most_freq_least.index[0]
    count_top = sum(distr_top.values)
    count_least = sum(distr_least.values)
    
    # Numeric variables considered only
    if type(value_top) != str:
        ratio = value_top / value_least
        record[column]=(ratio, value_top, value_least)
    
num_factors = 6
record_pd = pd.DataFrame(record, index=['ratio', 'value_top', 'value_least'])
factors = record_pd.loc['ratio'].nlargest(num_factors)
# factors.drop(['ConvertedCompYearly','CompTotal'], inplace=True)
print(f"\nThe major difference between the Top Earners and Least Earners is: {factors.head(1).index[0]}")
print(f"-Top Earners have a {factors.head(1).index[0]},that is {factors.head(1)[0]} that of the Least Earners.")





# Extra Analysis 1
# -----------------------------------------------------------------------------
print("\n\n")
print_bar()
print("Top Young Earning Respondents")
print_bar()

print("Who are the top young earners, (Age: 25-34') and what do they have in common : ")
new_mask = data_frame['Age']=='25-34'

young_earners = data_frame[new_mask.reindex_like(data_frame)]
new_mask = young_earners['Country'].isin(L)
young_model = young_earners#[new_mask.reindex_like(young_earners)]
filter_ =  "ConvertedCompYearly"
youngtop_x_ids = young_model[filter_].nlargest(x).index.tolist()
youngtop_earners =  data_frame.iloc[youngtop_x_ids]

for column in column_names:
    distr = youngtop_earners[column].value_counts()
    xmost_freq = distr.head(1)
    indices = xmost_freq.index
    values = xmost_freq.values
    count = sum(distr.values)
    max_freq = distr.max()
    print(f'\n{column}:')
    for index in range(len(indices)): 
        print(f'{indices[index]} -------- {round((values[index]/count) * 100, 2)}%')

        

# Extra Analysis 2
# -----------------------------------------------------------------------------
print("\n\n")
print_bar()
print("Abnormal Earnings - Case Study")
print_bar()

# Alexandr Wang?
hist_data = top_earners[filter_]
max_income = hist_data.value_counts().index.max()
max_income_df = data_frame.iloc[hist_data.idxmax()]
respondent_id = hist_data.idxmax()
print(f"\nAppearance of an outlier / abnormal data : ${format(int(max_income), ',')} -- Respondent : {respondent_id}")
print(f"Age : {data_frame.loc[respondent_id]['Age']}")
print(f"CompTotal : {data_frame.loc[respondent_id]['CompTotal']}")
print(f"Country : {data_frame.loc[respondent_id]['Country']}")
print(f"Currency : {data_frame.loc[respondent_id]['Currency']}")
print(f"DevType : {data_frame.loc[respondent_id]['DevType']}")
print(f"EdLevel : {data_frame.loc[respondent_id]['EdLevel']}")
print(f"WorkExp : {data_frame.loc[respondent_id]['WorkExp']}")
print(f"PurchaseInfluence : {data_frame.loc[respondent_id]['PurchaseInfluence']}")
print(f"RemoteWork : {data_frame.loc[respondent_id]['RemoteWork']}")
print(f"ResponseId : {data_frame.loc[respondent_id]['ResponseId']}")
print(f"YearsCode : {data_frame.loc[respondent_id]['YearsCode']}")
print(f"YearsCodePro : {data_frame.loc[respondent_id]['YearsCodePro']}")
print(f"DatabaseWantToWorkWith : {data_frame.loc[respondent_id]['DatabaseWantToWorkWith']}")


# Extra Analysis 3
# -----------------------------------------------------------------------------
print("\n\n")
print_bar()
print("No Degree Earnings - Case Study")
print_bar()
filter_  = 'no_degree'
mask = data_frame['EdLevel'] == filter_
drop_outs = data_frame[mask.reindex_like(data_frame)]
filter_ =  "ConvertedCompYearly"
no_degree_top_x_ids = drop_outs[filter_].nlargest(x).index.tolist()
no_degree_top_earners =  data_frame.iloc[no_degree_top_x_ids]

for column in column_names:
    distr = no_degree_top_earners[column].value_counts()
    xmost_freq = distr.head(1)
    indices = xmost_freq.index
    values = xmost_freq.values
    count = sum(distr.values)
    max_freq = distr.max()
    print(f'\n{column}:')
    for index in range(len(indices)): 
        print(f'{indices[index]} -------- {round((values[index]/count) * 100, 2)}%')
# -----------------------------------------------------------------------------
# Model to predict Income
# -----------------------------------------------------------------------------
lm_model = LinearRegression()

# - Model Evaluation
# -----------------------------------------------------------------------------
print("\n\n")
print_bar()
print("Predicting Annual Compensation from Work Experience - Model Evaluation")
print_bar()
y_preds, y_test, x_test, r2_val, mse = data_eval(
    lm_model, prepared_data, output_var)
print(
    f"\nThe r-squared score and mean squared error value for the model using only quant variables are {round(r2_val, 2)} and {mse}, respectively on {len(y_test)} values.")
print_bar()
y_preds = pd.Series(y_preds, name="y_preds")

x_label = input_var[2]
y_label = output_var[1]

x_test = x_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# concatinating the data frames to enable sorting by x-label
df_eval = pd.concat([x_test, y_test, y_preds], axis=1)

df_eval_sorted = df_eval.sort_values(by=[x_label])
x_vals = df_eval_sorted[x_label]
y_preds = df_eval_sorted['y_preds']
y_vals = df_eval_sorted[y_label]

x_vals_ = []
y_vals_ = []
x_vals_.append(x_vals)
x_vals_.append(x_vals)
y_vals_.append(y_vals)
y_vals_.append(y_preds)

# plt.show()
data_plot(x_label, y_label, x_vals_, y_vals_, save_dir+"predictive-model.png")

# - Evaluation
# dataModel.fit(x,y)
# data_plot(x,y)
# -----------------------------------------------------------------------------





 # for index_x in range(0, len(input_var)):
 #     for index_y in range(0, len(output_var)):
 #         x_label = input_var[index_x]
 #         y_label = output_var[index_y]
 #         sorted_data = prepared_data.sort_values(by=[x_label])

 #         x_vals = list(sorted_data[x_label])[:]
 #         y_vals = list(sorted_data[y_label])[:]

 #         data_plot(x_label, y_label, x_vals, y_vals)