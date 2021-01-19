from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from typing import List
import pandas as pd


def visualize_dataset_attributes(df):
    """
    Plots all the column of df
    """
    for column in df.columns:
        figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(df[column])
        plt.title(f'Values of the Column With Index {column}')
        plt.ylabel("Attribute Value")
        plt.show()


def choose_divider(test_type: str):
    """
    Depending on the test_type returns 1, 100, 1000,
    which is then used to divide testing value in a loop.
    The function is needed because in range() function
    there is no possibility to increase values for non-integer values.
    """
    if test_type == 'map_size':
        return 1
    if test_type == 'n_iter':
        return 1
    if test_type == 'learning_rate':
        return 1000
    if test_type == 'sigma':
        return 100


def make_single_plot(parameter_range: tuple, test_type: str, data_train: List[float], data_test: List[float],
                     title: str, xlabel: str, ylabel: str) -> None:
    """
    Make single plot of the metric tested in a project.
    """
    figure(num=None, figsize=(10, 5), dpi=80, facecolor='w', edgecolor='k')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # create x axis
    divider = choose_divider(test_type)
    range_list = [e / divider for e in list(range(parameter_range[0], parameter_range[1], parameter_range[2]))]

    plt.plot(range_list, data_train, label='train')
    plt.plot(range_list, data_test, label='test')
    plt.legend(loc="upper left")
    filename = title.replace(' ', '_').lower()
    plt.savefig(f'../images/{test_type}_{filename}.png')
    plt.show()


def choose_plot_title(test_type: str):
    """
    Returns proper image title according to the test_type
    """
    if test_type == 'map_size':
        return 'Number of Assigned Clusters vs Map Size'
    if test_type == 'n_iter':
        return 'Number of Assigned Clusters vs Number of Training Iterations'
    if test_type == 'learning_rate':
        return 'Number of Assigned Clusters vs Learning Rate'
    if test_type == 'sigma':
        return 'Number of Assigned Clusters vs Sigma'


def choose_xlabel(test_type: str):
    """
    Returns proper xlabel according to the test_type
    """
    if test_type == 'map_size':
        return 'Root of Map Clusters'
    if test_type == 'n_iter':
        return 'n_iter'
    if test_type == 'learning_rate':
        return 'learning_rate'
    if test_type == 'sigma':
        return 'sigma'


def plot_summary(test_type: str, parameter_range: tuple, len_df_profit_per_cluster_train_list: List[float],
                 len_df_profit_per_cluster_test_list: List[float],
                 buy_clusters_mean_profit_train_list: List[float],
                 sell_clusters_mean_profit_train_list: List[float],
                 buy_clusters_mean_profit_test_list: List[float],
                 sell_clusters_mean_profit_test_list: List[float]) -> None:
    """
    Plots summary of the training process.
    Arguments:
        test_type: type of the test, can be 'map_size', 'learning_rate', 'n_iter' or 'sigma'
        parameter_range: range in which the parameter will be tested, for example (1,100,1)
        len_df_profit_per_cluster_train_list: list of assigned clusters in training dataset
        len_df_profit_per_cluster_test_list: list of assigned clusters in testing dataset
        buy_clusters_mean_profit_train_list: list of mean profits in buy class for training data
        sell_clusters_mean_profit_train_list: list of mean profits in sell class for training data
        buy_clusters_mean_profit_test_list: list of mean profits in buy class for testing data
        sell_clusters_mean_profit_test_list: list of mean profits in sell class for testing data
    """
    title = choose_plot_title(test_type)
    xlabel = choose_xlabel(test_type)
    make_single_plot(parameter_range=parameter_range,
                     test_type=test_type,
                     data_train=len_df_profit_per_cluster_train_list,
                     data_test=len_df_profit_per_cluster_test_list,
                     title=title,
                     xlabel=xlabel,
                     ylabel='Assigned Clusters')

    make_single_plot(parameter_range=parameter_range,
                     test_type=test_type,
                     data_train=buy_clusters_mean_profit_train_list,
                     data_test=buy_clusters_mean_profit_test_list,
                     title='Buy Cluster Mean Profit',
                     xlabel=xlabel,
                     ylabel='Mean Profit')

    make_single_plot(parameter_range=parameter_range,
                     test_type=test_type,
                     data_train=sell_clusters_mean_profit_train_list,
                     data_test=sell_clusters_mean_profit_test_list,
                     title='Sell Cluster Mean Profit',
                     xlabel=xlabel,
                     ylabel='Mean Profit')


# def plot_avg_cluster_profit(df: pd.core.frame.DataFrame) -> None:
#     """
#
#     """
#     figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
#
#     x = list(df.groupby(by='cluster')['profit'].mean().sort_values())
#     plt.plot([10] * len(x), c='g')
#     plt.plot([-10] * len(x), c='r')
#     plt.plot(x)
#
#     plt.title("Average Profit Per Cluster", fontsize=25)
#     plt.ylabel("Profit", fontsize=20)
#     plt.xlabel("Index", fontsize=20)
#
#     plt.show()


def plot_strategy(df: pd.core.frame.DataFrame, dataset: str, buy_cluster: List[int],
                  sell_cluster: List[int]) -> None:
    """
    Plots stock prices ordered by the date and marks buy and sell
    moments returned by an algorithm.

    Arguments:
        df: ordered by date dataframe containing cluster and Price columns
        dataset: type of dataset - in our case Training or Testing
        buy_cluster: list of clusters assigned to buy class
        sell_cluster: list of clusters assigned to sell class
    """
    figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(df['Price'], color='lightblue', marker='o', markeredgecolor='green',
             markevery=list(df.loc[df['cluster'].isin(buy_cluster)].index))
    plt.plot(df['Price'], color='lightblue', marker='o', markeredgecolor='red',
             markevery=list(df.loc[df['cluster'].isin(sell_cluster)].index))
    plt.title(f"Investment Strategy for {dataset} Dataset", fontsize=25)


def plot_correlation_matrix(df: pd.core.frame.DataFrame) -> None:
    """
    Plots correlation matrix from dataframe.
    """
    f = plt.figure(figsize=(10, 10))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title('Correlation Matrix', fontsize=16);
