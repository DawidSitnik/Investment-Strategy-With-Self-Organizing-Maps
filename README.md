# Investment Strategy With Self Organizing Kohonens Maps

### Aim of The Project
The purpose of the following work is to create an investment strategy operating on the historical data from the Polish stock exchange. Such a system may support decisions making by the human during investments on a stock market. 

The following problem will be solved by the usage of unsupervised clustering algorithm bases on the neural networks - **Self-organizing Kohonen Maps**. Its adventage comparing to the supervised algorithms, is that the unsupervised learning may provide better results, as no action is defined for the particular situation on the market. Instead, the most similar configurations of shares’ prices may be identified and labeled.

### System Description
Because the nature of each stock market company is different, it is not possible to build a single model for all of them. Therefore, the separate model will be trained for each company, as the particular enterprise is unique and its behavior in time does not depend on other companies.

The whole process consistes of two main phases which are: **data preparation** and **modeling**. During the first phase a set of parameters describing a company will be created in a form of timeseries dataset. Those parameters will be callculated from the available coefficients described in the next section. In the next phase, which is modeling, the similar groups of vectors (**clusters**) will be created, basing on dataset features in particular timestamp. Then, each cluster will be marked as **'buy'**, **'sell'** or **'hold'**. The labels will be assigned to the clusters according to their mean return i.e. the amount of money earned during the training data analysis. In high average return cases the 'buy' label will be assigned, in periods with 'low' revenue the 'low' label will be assigned and in period in which the return wasn't higher or lower than a certain treshold the 'hold' label will be assigned. The example figure shows the example of the time series for the daily closing shares’ prices.

<img src='images/buy_sell.png'>

On the next picture we can the corresponding 

Opis problemu, Dlaczego unsupervised moze mieć sens
Opis systemu
Jakie spółki wybraliśmy i dlaczego
Czym są mapy kohonena
Jakie zmienne tworzymy

