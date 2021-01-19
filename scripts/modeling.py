from typing import List

import pandas as pd
import numpy as np
from minisom import MiniSom
from data_prepare import get_data
from config import Config


def som_predict(x, som) -> int:
    """
    Predicts cluster basing on a data row and the model.
    Arguments:
        x: data row
        som: model
    Returns:
        cluster: number
    """
    result = som.winner(np.array(x))
    return 10 * result[0] + result[1]


def train_som(som_width: int, som_height: int, df: pd.core.frame.DataFrame, df_train: pd.core.frame.DataFrame,
              df_test: pd.core.frame.DataFrame, df_train_columns: pd.core.frame.DataFrame, n_iter: int, sigma=0.3,
              learning_rate=0.01):
    """
    Trains self-organizing map and returns train and test datasets with predicted clusters.
    Arguments:
        som_width - width of som map
        som_height - height of som map
        df - initially prepared dataset
        df_train - training dataset
        df_test - testing dataset
        df_train_columns - list of columns of training dataset
        n_iter - number of iteration during training
        sigma - sigma parameter for the model
        learning_rate - learning rate
    Returns:
        final_df_train - training dataset with predicted cluster
        final_df_test - testing dataset with predicted cluster
    """

    som = MiniSom(som_width, som_height, df_train.shape[1], sigma=sigma, learning_rate=learning_rate,
                  random_seed=0)
    som.train(df_train, n_iter)

    # converting numpy arrays to dataframes
    df_train = pd.DataFrame(df_train, columns=df_train_columns)
    df_test = pd.DataFrame(df_test, columns=df_train_columns)

    # creating column with cluster basing on model prediction
    df_train['cluster'] = df_train.apply(lambda x: som_predict(x, som), axis=1)
    df_test['cluster'] = df_test.apply(lambda x: som_predict(x, som), axis=1)

    # joining train and test dataframes with previously dropped columns, which will be useful in the further part of
    # the script
    final_df_train = df_train.join(df[['Date', 'Price', 'close_plus_20_days', 'profit']].iloc[:, :len(df_train)],
                                   lsuffix='_org')
    final_df_test = df_test.join(df[['Date', 'Price', 'close_plus_20_days', 'profit']].iloc[len(df_train):],
                                 lsuffix='_org')

    return final_df_train, final_df_test


def get_profit_per_cluster(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """
    Returns dataframe with mean profit per cluster basing on a df given as an argument
    Arguments:
        df: dataframe which contains cluster and profit columns
    """
    return pd.DataFrame(df.groupby(by='cluster')['profit'].mean(), columns=['profit']).reset_index()


def get_mean_profit_per_class_from_train_df(df_profit_per_cluster_train: pd.core.frame.DataFrame) -> tuple:
    """
    Basing on a dataframe given as an argument, returns
    mean profit per class (buy, sell) in training dataset.
    - sort dataframe descending by profit
    - marks 1/3 of clusters with the highest profit as buy
    - marks 1/3 of clusters with the lowest profit as sell
    - if data contains less than 3 different clusters returns AssertionError
    Arguments:
        df_profit_per_cluster_train: training data containing columns profit and cluster
    Returns:
        buy_clusters_mean_profit: float, mean profit in buy cluster
        buy_clusters_list: list of integers representing clusters marked as buy
        sell_clusters_mean_profit: float, mean profit in sell cluster
        sell_clusters_list: list of integers representing clusters marked as sell
    """
    # if condition returns False, AssertionError is raised:
    assert len(df_profit_per_cluster_train) >= 3, "Algorithm, returned less than 3 clusters."

    df_profit_per_cluster = df_profit_per_cluster_train.sort_values(by='profit', ascending=False)
    group_size = int(len(df_profit_per_cluster) / 3)

    buy_clusters_mean_profit = df_profit_per_cluster.iloc[:group_size]['profit'].mean()
    sell_clusters_mean_profit = df_profit_per_cluster.iloc[-group_size:]['profit'].mean()

    buy_clusters_list = list(df_profit_per_cluster.iloc[:group_size]['cluster'])
    sell_clusters_list = list(df_profit_per_cluster.iloc[-group_size:]['cluster'])

    return buy_clusters_mean_profit, buy_clusters_list, sell_clusters_mean_profit, sell_clusters_list


def get_mean_profit_per_class_from_test_df(df_profit_per_cluster_test: pd.core.frame.DataFrame,
                                           buy_clusters_list: List[int], sell_clusters_list: List[int]) -> tuple:
    """
    Basing on a dataframe given as an argument, and list of buy and sell clusters
    returns mean profit per class (buy, sell) in testing dataset.
    Arguments:
        df_profit_per_cluster_test: testing data containing columns profit and cluster
        buy_clusters_list: list of buy clusters
        sell_clusters_list: list of sell clusters
    Returns:
        buy_clusters_mean_profit: float, mean profit in buy cluster
        sell_clusters_mean_profit: float, mean profit in sell cluster
        """
    # if condition returns False, AssertionError is raised:
    assert len(buy_clusters_list) != 0 and len(sell_clusters_list) != 0, "Clusters list can't be empty."

    buy_clusters_mean_profit = \
        df_profit_per_cluster_test.loc[df_profit_per_cluster_test['cluster'].isin(buy_clusters_list)]['profit'].mean()
    sell_clusters_mean_profit = \
        df_profit_per_cluster_test.loc[df_profit_per_cluster_test['cluster'].isin(sell_clusters_list)]['profit'].mean()

    return buy_clusters_mean_profit, sell_clusters_mean_profit


def create_final_strategy(filename: str, columns_list: List[str], som_width=Config.som_width,
                          som_height=Config.som_height, n_iter=Config.n_iter, sigma=Config.sigma,
                          learning_rate=Config.learning_rate) -> tuple:
    """
    Used for creating a final strategy (not for testing)
    - reads preprocessed split into training and testing sets data
    - train som model
    - calculates mean profit per cluster in training dataset
    - gets list of sell and buy clusters
    Arguments:
        filename: name of file with data
        columns_list: list of columns which should be left in the training data
        som_width: width of som map
        som_height: height of som map
        n_iter: number of iterations in som map
        sigma: sigma parameter for som map
        learning_rate: learning rate for som map
    Returns:
        final_df_train: training dataset
        final_df_test: testing dataset
        buy_clusters_list: list of buy clusters
        sell_clusters_list: list of sell clusters
    """

    print(
        f'Creating final strategy for parameters: \nmap_size: {som_height}\nn_iter: {n_iter}\nsigma:{sigma}\nlr: {learning_rate}')
    # get prepared data
    df, df_prepared, df_train, df_test, df_train_columns = get_data(filename, columns_list)

    # train som
    final_df_train, final_df_test = train_som(som_width, som_height, df, df_train, df_test, df_train_columns, n_iter,
                                              sigma=sigma, learning_rate=learning_rate)

    # get profit per cluster in train datasets
    df_profit_per_cluster_train = get_profit_per_cluster(final_df_train)

    assert len(df_profit_per_cluster_train) >= 3, "Algorithm, returned less than 3 clusters."

    df_profit_per_cluster = df_profit_per_cluster_train.sort_values(by='profit', ascending=False)
    group_size = int(len(df_profit_per_cluster) / 3)

    buy_clusters_list = list(df_profit_per_cluster.iloc[:group_size]['cluster'])
    sell_clusters_list = list(df_profit_per_cluster.iloc[-group_size:]['cluster'])

    return final_df_train, final_df_test, buy_clusters_list, sell_clusters_list


def create_strategy(filename: str, columns_list: List[str], som_width: int, som_height: int, n_iter: int, sigma=0.3,
                    learning_rate=0.01) -> tuple:
    """
    Creates strategy which can be used in testing part of the script.
    - reads preprocessed split into training and testing sets data
    - train som model
    - calculates mean profit per cluster in training and testing dataset
    - gets mean profits
    Arguments:
        filename: name of file with data
        columns_list: list of columns which should be left in the training data
        som_width: width of som map
        som_height: height of som map
        n_iter: number of iterations in som map
        sigma: sigma parameter for som map
        learning_rate: learning rate for som map
    Returns:
        len(df_profit_per_cluster_train): amount of used clusters in training data
        len(df_profit_per_cluster_test): amount of used clusters in testing data
        buy_clusters_mean_profit_train: mean profit in buy clusters for training data
        sell_clusters_mean_profit_train: mean profit in sell clusters for training data
        buy_clusters_mean_profit_test: mean profit in buy clusters for testing data
        sell_clusters_mean_profit_test: mean profit in sell clusters for testing data
    """
    # get prepared data
    df, df_prepared, df_train, df_test, df_train_columns = get_data(filename, columns_list)

    # train som
    final_df_train, final_df_test = train_som(som_width, som_height, df, df_train, df_test, df_train_columns, n_iter,
                                              sigma=sigma, learning_rate=learning_rate)

    # get profit per cluster in train and test datasets
    df_profit_per_cluster_train = get_profit_per_cluster(final_df_train)
    df_profit_per_cluster_test = get_profit_per_cluster(final_df_test)

    # get mean profit for sell and buy class in training and testing datasets
    try:
        buy_clusters_mean_profit_train, buy_clusters_list, sell_clusters_mean_profit_train, sell_clusters_list = \
            get_mean_profit_per_class_from_train_df(df_profit_per_cluster_train)

        buy_clusters_mean_profit_test, sell_clusters_mean_profit_test = \
            get_mean_profit_per_class_from_test_df(df_profit_per_cluster_test, buy_clusters_list, sell_clusters_list)
    # if the data was assigned to less than to 3 clusters
    except:
        buy_clusters_mean_profit_train, sell_clusters_mean_profit_train, \
        buy_clusters_mean_profit_test, sell_clusters_mean_profit_test = None, None, None, None

    return len(df_profit_per_cluster_train), len(df_profit_per_cluster_test), \
           buy_clusters_mean_profit_train, sell_clusters_mean_profit_train, \
           buy_clusters_mean_profit_test, sell_clusters_mean_profit_test


def set_model_parameters(test_type: str, parameter_value: float):
    """
    Returns parameters which should be used in one iteration
    of the testing process.
    """
    # assigning default parameters for the model
    som_width = Config.som_width
    som_height = Config.som_height
    n_iter = Config.n_iter
    sigma = Config.sigma
    learning_rate = Config.learning_rate

    # assign testing parameter to the model parameter basing on test_parameter value
    if test_type == 'map_size':
        som_width = parameter_value
        som_height = parameter_value
    if test_type == 'n_iter':
        n_iter = parameter_value
    if test_type == 'learning_rate':
        learning_rate = parameter_value / 1000
    if test_type == 'sigma':
        sigma = parameter_value / 100
    return som_width, som_height, n_iter, sigma, learning_rate


def test_som_parameters(filename: str, columns_list: List[str], test_type: str, parameter_range: tuple) -> tuple:
    """
    Tests parameters of som:
    - initializes empty lists for measures metrics
    - in each loop iteration creates strategy and append calculated metrics to the proper lists
    Arguments:
         filename: name of file from the data is loaded
         columns_list: list of column for training dataset
         test_type: type of testing, can be 'map_size', 'learning_rate', 'n_iter', 'sigma'
         parameter_range: range in which parameters will be tested
    Returns:
        len_df_profit_per_cluster_train_list: list of amounts of used clusters in training data
        len_df_profit_per_cluster_test_list: list of amounts of used clusters in testing data
        buy_clusters_mean_profit_train_list: list of mean profit sin buy clusters for training data
        sell_clusters_mean_profit_train_list: list of mean profits in sell clusters for training data
        buy_clusters_mean_profit_test_list: list of mean profits in buy clusters for testing data
        sell_clusters_mean_profit_test_list: list of mean profits in sell clusters for testing data
    """
    assert test_type in ['map_size', 'learning_rate', 'n_iter',
                         'sigma'], "test_type must be from a list ['map_size', 'learning_rate', 'n_iter', 'sigma']"

    # declare empty lists
    len_df_profit_per_cluster_train_list, len_df_profit_per_cluster_test_list, \
    buy_clusters_mean_profit_train_list, sell_clusters_mean_profit_train_list, \
    buy_clusters_mean_profit_test_list, sell_clusters_mean_profit_test_list = [], [], [], [], [], []

    # testing loop
    for testing_parameter in range(parameter_range[0], parameter_range[1], parameter_range[2]):
        som_width, som_height, n_iter, sigma, learning_rate = set_model_parameters(test_type, testing_parameter)

        # make predictions
        len_df_profit_per_cluster_train, len_df_profit_per_cluster_test, \
        buy_clusters_mean_profit_train, sell_clusters_mean_profit_train, \
        buy_clusters_mean_profit_test, sell_clusters_mean_profit_test = \
            create_strategy(filename, columns_list, som_width, som_height, n_iter, sigma=sigma,
                            learning_rate=learning_rate)

        # appending parameters to proper lists
        len_df_profit_per_cluster_train_list.append(len_df_profit_per_cluster_train)
        len_df_profit_per_cluster_test_list.append(len_df_profit_per_cluster_test)
        buy_clusters_mean_profit_train_list.append(buy_clusters_mean_profit_train)
        sell_clusters_mean_profit_train_list.append(sell_clusters_mean_profit_train)
        buy_clusters_mean_profit_test_list.append(buy_clusters_mean_profit_test)
        sell_clusters_mean_profit_test_list.append(sell_clusters_mean_profit_test)

    return len_df_profit_per_cluster_train_list, len_df_profit_per_cluster_test_list, \
           buy_clusters_mean_profit_train_list, sell_clusters_mean_profit_train_list, \
           buy_clusters_mean_profit_test_list, sell_clusters_mean_profit_test_list
