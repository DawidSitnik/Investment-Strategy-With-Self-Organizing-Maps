# Investment Strategy With Self Organizing Kohonens Maps

### Aim of The Project
The purpose of the following work is to create an investment strategy operating on the historical data from the Polish stock exchange. Such a system may support decisions making by the human during investments on a stock market. 

The following problem will be solved by the usage of unsupervised clustering algorithm bases on the neural networks - **Self-organizing Kohonen Maps**. Its adventage comparing to the supervised algorithms, is that the unsupervised learning may provide better results, as no action is defined for the particular situation on the market. Instead, the most similar configurations of shares’ prices may be identified and labeled.

### System Description
Because the nature of each stock market's company is different, it is not possible to build a single model for all of them. Therefore, the separate model will be trained for each company, as the particular enterprise is unique and its behavior in time does not depend on the other companies.

The whole process consistes of two main phases which are: **data preparation** and **modeling**. During the first phase a set of parameters describing a company will be created in a form of timeseries dataset. Those parameters will be callculated from the available coefficients described in the next section. In the next phase, which is modeling, the similar groups of vectors (**clusters**) will be created, basing on dataset features in particular timestamp. Then, each cluster will be marked as **'buy'**, **'sell'** or **'hold'**. The labels will be assigned to the clusters according to their mean return value. At the picture bellow we can see a good moments for selling or buying a stock:

<img src='images/buy_sell.png'>

According to the tresholds due to which we set clusters labels, a strategy can be more or less agresive. Tresholds don't need to be symetric as they reflect the
character of the investor and his inclination towards taking the risk.

The whole 




Opis problemu, Dlaczego unsupervised moze mieć sens
Opis systemu
Jakie spółki wybraliśmy i dlaczego
Czym są mapy kohonena
Jakie zmienne tworzymy

