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

# Data Generation 
coef = np.array([0.1, 0.9, -1])
arparams = [np.array([1,]), np.array([0.5]), np.array([-0.8])]
maparams = [np.array([0.1,]), np.array([0.5]), np.array([0.1])]
population = np.array([60, 20, 20])

df, x = data.data_generation(coef, arparams, maparams, population)
df["Exogenous variable"] = x
time = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
os.makedirs("Project/data/"+time+"/")
df.to_csv("Project/data/"+time+"/generated_series.csv")
del df["Exogenous variable"]

# Variables 
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

run = True

if run :
    S, Theta, W = samplers.Gibbs(df, innit, K, models, params, epochs=10**1)
    os.makedirs("Project/output/model_"+time+"/")
    np.save("Project/output/model_"+time+"/States.npy", S)
    np.save("Project/output/model_"+time+"/Theta.npy", Theta)
    np.save("Project/output/model_"+time+"/Weights.npy", W)

    labels = models[0].fit().param_names
    graphs.graph(S, Theta, W, labels, dir="Project/output/model_"+time+"/")