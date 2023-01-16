import pandas as pd
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import os 
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

import samplers
import data
import graphs

def main_(model_name=1, epochs=5*10**3) : 

    if model_name == 3 : 
        coef = np.array([0.1, 0.9, -1])
        arparams = [np.array([1,]), np.array([0.5]), np.array([-0.8])]
        maparams = [np.array([0.1,]), np.array([0.5]), np.array([0.1])]
        population = np.array([60, 20, 20])
    elif model_name == 1 : 
        coef = np.array([0, 0])
        arparams = [np.array([1,]), np.array([0.2]),]
        maparams = [np.array([0.5,]), np.array([0.9]),]
        population = np.array([70, 30])
    else : 
        coef = np.array([0.1, 0.9, -1])
        arparams = [np.array([1, -0.5]), np.array([-0.3, 0.3]), np.array([0.2, -0.3])]
        maparams = [np.array([0.1, 0.2,]), np.array([0.1, -0.1]), np.array([0.1, -0.2])]
        population = np.array([60, 20, 20])


    df, x = data.data_generation(coef, arparams, maparams, population)
    df["Exogenous variable"] = x
    time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    os.makedirs("data/model_"+str(model_name)+"/"+time)
    df.to_csv("data/model_"+str(model_name)+"/"+time+"/generated_series.csv")
    del df["Exogenous variable"]

    # Variables
    if model_name == 1 : 
        K = 2
        ARIMA_order = (1, 0, 1)
        args_model = {
        "enforce_stationarity":False, 
        "order":ARIMA_order, 
        "trend":'n',
        }
    elif model_name == 2 : 
        K = 3
        ARIMA_order = (2, 0, 2)
        args_model = {
        "enforce_stationarity":False, 
        "order":ARIMA_order, 
        "trend":'n',
        }
    else : 
        K = 3
        ARIMA_order = (1, 0, 1)
        args_model = {
            "enforce_stationarity":False, 
            "order":ARIMA_order, 
            "trend":'n',
            "exog":x,
            }

    models = {col : ARIMA(endog=np.array(df[col]), **args_model) for col in df.columns}
    params = {col : models[col].fit().params for col in df.columns}
    innit = samplers.initialization(K, df, params)


    S, Theta, W = samplers.Gibbs(df, innit, K, models, params, epochs=epochs)
    os.makedirs("output/model_"+str(model_name)+"/"+time+"/")
    np.save("output/model_"+str(model_name)+"/"+time+"/States.npy", S)
    np.save("output/model_"+str(model_name)+"/"+time+"/Theta.npy", Theta)
    np.save("output/model_"+str(model_name)+"/"+time+"/Weights.npy", W)

    labels = models[0].fit().param_names
    graphs.graph(S, Theta, W, labels, dir="output/model_"+str(model_name)+"/"+time+"/")

run = False 

if run : 
    #main_(1)
    #main_(2)
    #main_(3)
    None 