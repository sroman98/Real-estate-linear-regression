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
    for param, i in zip(params, range(len(params))):
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

# load data
df_cols = ["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude","Y house price of unit area"]
og_df = pd.read_csv("Real estate.csv", usecols=df_cols)

# scale data
scaler = MinMaxScaler()
scaler.fit(og_df)
df = pd.DataFrame(scaler.transform(og_df), columns=df_cols)

# divide data into x and y
x_cols = ["X1 transaction date","X2 house age","X3 distance to the nearest MRT station","X4 number of convenience stores","X5 latitude","X6 longitude"]
df_x = df[x_cols]
df_y = df["Y house price of unit area"]

# add x attribute for bias
df_x["X0 bias"] = 1

# use 20% of the data for testing
test_size = -1 * math.floor(df_y.size * 0.2)

# split the data into training/testing sets
x_train = df_x[:test_size]
x_test = df_x[test_size:]
y_train = df_y[:test_size]
y_test = df_y[test_size:]

# training
epoch = 0
hyps = []

# init all params with 0
params = []
for _ in range(len(x_train.columns)):
    params.append(0)

# calculate initial hypotheses and mse
hyps = calculate_hypotheses(params,x_train)
mse = mean_squared_error(y_train,hyps)

# use gradient descent until 5000 epochs are reached or mse is 0
while epoch < 1000 and mse > 0:
    old_params = params
    params = gradient_descent(params, 0.2, y_train, hyps, x_train)
    hyps = calculate_hypotheses(params,x_train)
    mse = mean_squared_error(y_train,hyps)
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
    print(x_train.columns[i] + ": %.4f" % params[i])

test_hyps = calculate_hypotheses(params,x_test)
print('Coefficient of determination: \n %.2f' % r2_score(y_test, test_hyps))

# Predictions with inputs
print("\nPREDICT HOUSE PRICE OF UNIT AREA BASED ON TAIWAN DATA")

predict = "Y"

while predict == "Y" or predict == "y":
    date = float(input("Transaction date: "))
    age = float(input("House age: "))
    distance = float(input("Distance to the nearest MRT station: "))
    stores = int(input("Number of convenience stores in the living circle on foot: "))
    latitude = float(input("Latitude: "))
    longitude = float(input("Longitude: "))

    # Create data frame with predictions
    x_pred = pd.DataFrame([(date, age, distance, stores, latitude,longitude,0)], columns = df_cols)
    x_pred = pd.DataFrame(scaler.transform(x_pred), columns=df_cols)
    x_pred = x_pred[x_cols]
    x_pred["X0 bias"] = 1

    pred = calculate_hypotheses(params,x_pred)
    pred_df = pd.DataFrame([(date, age, distance, stores, latitude,longitude,pred[0])], columns = df_cols)
    pred_df = scaler.inverse_transform(pred_df)

    print("\nHouse price of unit area: %f (10,000 New Taiwan Dollar/Ping)" %pred_df[0][6])

    predict = input("Do you want to make another prediction? (Y/N): ")
    print("")
