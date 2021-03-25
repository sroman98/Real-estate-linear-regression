import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import math
from sklearn.metrics import r2_score

def mean_squared_error(ys, hyps):
    acum = 0
    for y, hyp in zip(ys, hyps):
        res = (y - hyp) ** 2
        acum += res
    return acum / len(ys)

def gradient_descent_summatory(ys, hyps, xs):
    acum = 0
    for y, hyp, x in zip(ys, hyps, xs):
        res = (hyp - y) * x
        acum += res
    return acum

def gradient_descent(params, lr, ys, hyps, xmat):
    new_params = []
    m = len(ys)
    cols = xmat.columns
    for param, i in zip(params,range(len(params))):
        res = param - (lr/m) * gradient_descent_summatory(ys, hyps, xmat[cols[i]])
        new_params.append(res)
    return new_params

def calculate_hypotheses(params, xmat):
    hyps = []
    cols = len(xmat.columns)
    rows = len(xmat)
    for j in range(rows):
        hyp = 0
        for i in range(cols):
            hyp += xmat.iloc[j, i] * params[i]
        hyps.append(hyp)
    return hyps

df_cols = ["X1","X2","Y"]
df = pd.read_csv("ex.csv", usecols=df_cols)

# divide data into x and y
x_cols = ["X1","X2"]
df_x = df[x_cols]
df_y = df["Y"]

# add x attribute for bias
df_x["X0"] = 1

# training
epoch = 0
hyps = []

# init all params with 0
params = [0,0,8]

# calculate initial hypotheses and mse
hyps = calculate_hypotheses(params,df_x)
mse = mean_squared_error(df_y,hyps)

# use gradient descent until 5000 epochs are reached or mse is 0
while epoch < 2 and mse > 0:
    old_params = params
    params = gradient_descent(params, 1, df_y, hyps, df_x)
    hyps = calculate_hypotheses(params,df_x)
    mse = mean_squared_error(df_y,hyps)
    epoch += 1
    # break if parameters weren't updated
    if old_params == params:
        break

print("LR MODEL INFO")
print("Mean squared error:")
print(mse)
print("Epochs:")
print(epoch)
print("Parameters:")
for i in range(len(params)):
    print(df_x.columns[i] + ": " + str(params[i]))
