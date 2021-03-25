import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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
    for param, i in zip(params, range(m)):
        res = param - (lr/m) * gradient_descent_summatory(ys, hyps, xmat.iloc[i])
        new_params.append(res)
    return new_params

def calculate_hypotheses(params, xmat):
    hyps = []
    cols = len(xmat.columns)
    for j in range(len(xmat)):
        hyp = 0
        for i in range(cols):
            hyp += df.iloc[j, i] * params[i]
        hyps.append(hyp)
    return hyps


MAX_EPOCHS = 5000
epoch = 0

hyps = []

og_df = pd.read_csv("Real estate.csv", usecols=["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude","Y house price of unit area"])
scaler = MinMaxScaler()
scaler.fit(og_df)
df = pd.DataFrame(scaler.transform(og_df), columns=["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude","Y house price of unit area"])

df_x = df[["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude"]]
df_x["X0 bias"] = 1

df_y = df["Y house price of unit area"]

params = []
for _ in range(len(df_x.columns)):
    params.append(1)
hyps = calculate_hypotheses(params,df_x)
mse = mean_squared_error(df_y,hyps)

while epoch < MAX_EPOCHS and mse > 0:
    old_params = params
    params = gradient_descent(params, 0.04, df_y, hyps, df_x)
    hyps = calculate_hypotheses(params,df_x)
    mse = mean_squared_error(df_y,hyps)
    epoch += 1
    if old_params == params:
        break

print("ERROR")
print(mse)
print("EPOCHS")
print(epoch)
print("PARAMS")
print(params)