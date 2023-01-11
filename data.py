import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.stats import multinomial
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from itertools import chain
import random as rd
from tqdm import tqdm
import statsmodels as sm
from scipy.stats import norm
from scipy.stats import dirichlet

def data_generation(coef, arparams, maparams, population):
    """
    Generates data with the specificated values
    Input :
        - coefs : coefficients with exogenous variable
        - arparams : ar coefficients for each clusters
        - maparams : ma coefficients for each clusters 
        - population : number of time serie per cluster 
    Output :
        - df : a dataframe with time series 
        - x : the exogenous variable 
    """

    data = pd.DataFrame([t for t in range(population.sum().astype(int))])

    # Exogenous variable generation 
    ar = np.array([1,])
    ma = np.array([0.1,])
    ar = np.r_[1, -ar] # add zero-lag and negate
    ma = np.r_[1, ma] # add zero-lag
    x = sm.tsa.arima_process.arma_generate_sample(ar, ma, 100, scale=0.1, burnin=1000)

    # Clusters generation 
    for cluster in range(len(coef)): 

        t_innit = 0
        # Clusters coefficients 
        arparam = arparams[cluster]
        maparam = maparams[cluster]
        ar = np.r_[1, -arparam] 
        ma = np.r_[1, maparam] 

        for t in range(t_innit, t_innit+population[cluster]):
            data[t] = sm.tsa.arima_process.arma_generate_sample(
                ar, ma, 100, scale=0.1, burnin=1000
            )
            data[t] += coef[cluster]*x

    return(data, x)