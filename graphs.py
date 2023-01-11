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

def graph(S, Theta, W, labels, dir) : 
    """
    Create a graphic to illustrate the convergence of the algorithm
    Input :
        - S : the states 
        - Theta : the parameters coefficients 
        - W : the weights 
        - labels : the labels 
        - dir : directory to save to 
    Output 
        - a figure saved in the output directory
    """

    K = len(W[0])
    clusters = np.zeros((K, len(S)))
    for k in range(K):
        cluster = np.zeros(len(S))
        for t in range(len(S)):
            cluster[t] = ((S[t]==k).sum())
        clusters[k, :] = cluster

    fig, axes = plt.subplots(2, K, figsize=(7*K, 10), sharey=True)

    for k in range(K): 
        for i in range(len(labels)) : 
            sns.lineplot(
                x=range(1, len(Theta)+1), 
                y=Theta[:, k, i], 
                label=labels[i], 
                ax=axes[0, k]
            )
        axes[0, k].set_title(
            "Cluster "+str(k), 
            fontsize=14, 
            fontweight="bold"
        )
        axes[0, k].set_xscale("log")

    for cluster, k in zip(clusters, range(K)): 
        sns.lineplot(
            x=[t for t in range(1, len(S)+1)], 
            y=cluster, 
            ax=axes[1, k], 
            label="Proportion of cluster "+str(k)
            )
        axes[1, k].set_xscale("log")

    fig.suptitle("ARIMAX model coefficients", fontsize=16, fontweight="bold")

    fig.savefig(dir+"Model_results.pdf", format="pdf")