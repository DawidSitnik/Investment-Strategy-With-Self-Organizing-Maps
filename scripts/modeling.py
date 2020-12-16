from datetime import datetime
import pandas as pd


def read_csv(filename):
    df = pd.read_csv(f'../data/extracted_features/{filename}')
    df = df.drop(columns=['Unnamed: 0', 'f12']).dropna()
    df = df.rename(columns={' Close/Last':'Price'})
    return df


def prepare_df(df):
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

    # callculating profit in 20 days
    df = df.loc[df['close_plus_20_days'].notnull()].reset_index(drop=True)
    df['profit'] = df['close_plus_20_days'] - df['Price']

    return df