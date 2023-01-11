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

def conditionnal_sampler_1(theta, y, K, w, models, params, ARIMA_order) :
    """ 
    Run the first step in the Hasting algorithm iteration,
    we simulate from the S|θ distribution.
    
    Input : 
        - theta : the last generated theta variable, list of size K
        - y : the observations 
        - K : number of clusters 
        - w : a priori distribution for the clusters, if None then
    we assume a uniform distribution between clusters 
        - ARIMA_order : Arima model orders, for AR(p) use (p,0,0)
    
    Output :
        - S : the new generated S variable 
    """
    
    s, i = np.zeros(len(y.columns)), 0
    
    for col in y.columns :
        
        mod = models[col] 

        try :
            W = np.array([mod.loglike(theta[k], transformed=False) for k in range(K)])
            W = np.exp(W-W.max())*w
            W = np.nan_to_num(W)
            W = W/(W.sum())
            s[i] = np.random.choice(K, size=1, p=W)
        except : 
            print(w)
            W = np.array([mod.loglike(theta[k], transformed=False) for k in range(K)])
            print(W)
            W = np.exp(W-W.max())*w
            print(W)
            W = np.nan_to_num(W)
            print(W)
            W = W/(W.sum())
            print(W)
            s[i] = np.random.choice(K, size=1, p=W)
        i += 1
    return(s)

def conditionnal_sampler_2(theta_0, S, y, K, ARIMA_order, models, params, epochs=10) :
    """ 
    Run the second step in the Hasting algorithm iteration for AR(1)
    
    Input :
        - theta_0 : innitial value for theta paramters 
        - S : the last generated S variable
        - y : the observations
        - K : the number of clusters 
        - ARIMA_order : Arima model orders, for AR(p) use (p,0,0)
        -epochs : the number of iterations for the Metropolis Hasting algorithm
    
    Output : 
        - Theta : the new generated theta variable 
    """
    theta = theta_0.copy()
    # For each cluster 
    for k in range(K) : 
        # Run Metropolis Hasting algorithm 
        col_list = y.columns[S==k]
        # if no serie is in the cluster, we randomly select some
        if len(col_list)==0 : 
            col_list = y.sample(n=20).columns
        P = np.array(
                [models[col].loglike(theta[k], transformed=False)+norm.logpdf(theta[k], scale=0.3) for col in col_list],
                )
        P = P.sum() # We focus on the sum of the likelikehood 
        # Number of iterations 
        for t in range(epochs) : 
                
            noise = np.random.normal(size=len(theta[0]), scale=0.001)
            P_new = np.array(
                [models[col].loglike(theta[k]+noise, transformed=False)+norm.logpdf(theta[k]+noise, scale=0.3) for col in col_list],
                )
            P_new = P_new.sum()
            p = min(1, np.nan_to_num(np.exp(P_new - P)))
            s = np.random.binomial(n=1, p=p, size=1)
            if s == 1 :
                theta[k] += noise
                P = P_new
    return(theta)

def conditionnal_sampler_3(w, s, y, K, ARIMA_order, models, params, epochs=10) :
    """ 
    Run the second step in the Hasting algorithm iteration for AR(1)
    
    Input :
        - S : the last generated S variable
        - y : the observations
        - K : the number of clusters 
        - ARIMA_order : Arima model orders, for AR(p) use (p,0,0)
    
    Output : 
        - W : the a priori distribution for the clusters
    """
    # Run Metropolis Hasting algorithm 
    P = 1
    w = w + 0.01
    w = w / w.sum()
    # Initialisation 
    for i in range(len(s)) : 
        idx = s[i].astype(int)
        P = P*w[idx]*dirichlet.pdf(np.array([w[idx]]),np.array([4,4]))
    for t in range(epochs) : 
        noise = np.zeros(len(w))
        for i in range(len(w)) : 
            noise_ = np.random.normal(scale=0.001)
            while noise_ < - w[i] : 
                noise_ = np.random.normal(scale=0.001)
            noise[i] = noise_
        P_new = 1
        w_new = w+noise
        w_new = w_new / w_new.sum()
        for i in range(len(s)) : 
            idx = s[i].astype(int)
            P_new = P_new*w_new[idx]*dirichlet.pdf(np.array([w_new[idx]]),np.array([4,4]))
            p = min(1, np.nan_to_num(np.exp(P_new - P)))
            p = np.random.binomial(n=1, p=p, size=1)
            if p == 1 :
                w = w_new.copy()
                P = P_new

    return(w)

def Gibbs(y, innit, K, models, params, ARIMA_order=(1, 0, 0), epochs=1000,) : 
    """ 
    Run the Hasting algorithm 
    
    Input : 
        - y : the observations
        - K : the number of clusters
        - ARIMA_order : order of the ARIMA model
        - epochs : the number of iterations 
        
    Output : 
        - X, Y : list with added variables 
    """
    S = np.zeros((epochs, len(y.columns)))
    Theta = np.zeros((epochs, K,  len(params[0])))

    S[0, :], Theta[0, :, :] = innit[0], innit[1]
    args = {
        "y":y, 
        "K":K, 
        "ARIMA_order":ARIMA_order, 
        "models":models, 
        "params":params
        }
    
    W0 = np.ones(K) / K
    W = np.zeros((epochs, K))
    W[0] = W0

    for t in tqdm(range(1, epochs)) :

        S[t] = conditionnal_sampler_1(Theta[t-1], w=W[t-1], **args)
        Theta[t] = conditionnal_sampler_2(Theta[t-1], S[t-1], **args)
        W[t] = conditionnal_sampler_3(W[t-1], S[t-1], **args)
    return(S, Theta, W)

def initialization(K, df, params) : 
    """
    Initialize the states for the algorithms
    Input :
        - K : number of clusters
        - df : our generated data 
        - params : dictionnary with the parameters associated at each cluster 
    Output :
        - positions : Starting clusters
        - theta : Starting coefficients 
    """
    W0 = np.ones(K)/K
    
    ancestors = multinomial.rvs(n=len(df.columns), p=W0, size=1)[0]
    positions = [([idx]*ancestors[idx]) for idx in range(len(ancestors)) if ancestors[idx]>0]
    positions = list(chain(*positions))
    rd.shuffle(positions)
    positions = np.array(positions)
    
    theta = np.zeros((K, len(params[0])))
    for k in range(K) : 
        
        col_list = df.columns[positions==k]
        t = np.array([params[col] for col in col_list]).mean(axis=0)
        theta[k] = t
        #theta[k] = np.random.normal(size=len(params[0]), scale=0.1)
        
    return([positions, np.array(theta)])