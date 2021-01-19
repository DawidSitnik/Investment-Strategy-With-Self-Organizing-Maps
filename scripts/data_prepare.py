from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime


def variance_inflation_factors(exog_df:pd.core.frame.DataFrame) -> pd.core.series.Series:
    '''
    Design matrix with all explanatory variables, as for example used in regression.
    Arguments:
        exog_df - dataframe, (nobs, k_vars)

    Returns:
        vif : Series with vif per variable.
    '''
    exog_df = add_constant(exog_df)
    vifs = pd.Series(
        [1 / (1. - OLS(exog_df[col].values,
                       exog_df.loc[:, exog_df.columns != col].values).fit().rsquared)
         for col in exog_df],
        index=exog_df.columns,
        name='VIF'
    )
    return vifs


def read_csv(filename:str) -> pd.core.frame.DataFrame:
    """
    Reads certain csv file.

    Arguments:
        filename - name of csv file

    Returns:
        df - certain csv in a form of dataframe
    """
    df = pd.read_csv(f'../data/extracted_features/{filename}')
    df = df.drop(columns=['Unnamed: 0']).dropna()
    df = df.rename(columns={' Close/Last':'Price'})
    return df


def load_data_from_csv(filename:str) -> pd.core.frame.DataFrame:
    '''
    Loads data from certain csv file and makes some basics preprocessing.
    - converts date column which is in string into datetime,
    - sorts data ascending by date,
    - creates a column with price in 20 days,
    - calculates a profit

    Arguments:
        filename - name of csv file

    Returns:
        df - initially prepared dataset
    '''
    df = read_csv(filename)
    # converting str date to datetime
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y'))

    # sorting data ascending by date
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)

    # creating column with price in 20 days
    for i in range(1, len(df)):
        try:
            df.loc[i, 'close_plus_20_days'] = df.loc[i + 20, 'Price']
        except:
            df.loc[i, 'close_plus_20_days'] = None

    # calculating profit in 20 days
    df = df.loc[df['close_plus_20_days'].notnull()].reset_index(drop=True)
    df['profit'] = df['close_plus_20_days'] - df['Price']

    return df


def get_data(filename:str, columns_list=None) -> tuple:
    '''
    Getting data needed for modeling. Reads csv with prepared features,
    scales the data and selects specific column basing on columns_list.
    If columns_list is set to None, read all the columns.

    Arguments:
        filename - name of file with data
        columns_list - list of columns to read. If None reads all the columns.

    Returns:
        df - data from csv, after some basic preparation
        df_prepared - prepared data without a split into training and testing dataset
        df_train - training dataset
        df_test - testing dataset
        df_train_columns - list with training columns
    '''

    df = load_data_from_csv(filename)

    # dropping columns for training
    df_prepared = df.drop(columns=[' High', ' Low', 'profit', 'close_plus_20_days', 'Date'])
    df_prepared_columns = df_prepared.columns
    df_prepared = np.array(df_prepared)

    # scaling data
    scaler = MinMaxScaler()
    df_prepared = scaler.fit_transform(df_prepared)

    # leaving columns specified in columns_list
    if columns_list is not None:
        df_prepared = pd.DataFrame(df_prepared)[columns_list]
        df_prepared_columns = df_prepared.columns
    else:
        df_prepared = pd.DataFrame(df_prepared)

    # splitting dataset into training and testing
    train_len = len(df_prepared)-365
    df_train = np.array(df_prepared[:train_len])
    df_test = df_prepared[train_len:]

    return df, df_prepared, df_train, df_test, df_prepared_columns