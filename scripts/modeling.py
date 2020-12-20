from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def read_csv(filename):
    df = pd.read_csv(f'../data/extracted_features/{filename}')
    df = df.drop(columns=['Unnamed: 0']).dropna()
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


def som_predict(x, som):
    """
    Predicting cluster.
    """
    result = som.winner(np.array(x))
    return 10*result[0] + result[1]


def plot_avg_cluster_profit(df):
    figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')

    x = list(df.groupby(by='cluster')['profit'].mean().sort_values())
    plt.plot([10] * len(x), c='g')
    plt.plot([-10] * len(x), c='r')
    plt.plot(x)

    plt.title("Average Profit Per Cluster", fontsize=25)
    plt.ylabel("Profit", fontsize=20)
    plt.xlabel("Index", fontsize=20)

    plt.show()


def plot_strategy(df, dataset, buy_cluster, sell_cluster):
    figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(df['Price'],color='lightblue', marker='o', markeredgecolor='green', markevery=list(df.loc[df['cluster'].isin(buy_cluster)].index))
    plt.plot(df['Price'],color='lightblue', marker='o', markeredgecolor='red', markevery=list(df.loc[df['cluster'].isin(sell_cluster)].index))
    plt.title(f"Investment Strategy for {dataset} Dataset", fontsize=25)


def plot_corelation_matrix(np_array):
    df = pd.DataFrame(np_array)
    f = plt.figure(figsize=(19, 15))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
