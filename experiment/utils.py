import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

def plot(target, prediction, title):
    """
    Plot test set and prediction
    """
    plt.figure(figsize=(10, 6))
    plt.plot(target, label='Stock Price')
    plt.plot(prediction, label='Predicted Stock Price')
    plt.title(title)
    plt.xlabel('Time', fontsize=12, verticalalignment='top')
    plt.ylabel('Close', fontsize=14, horizontalalignment='center')
    plt.legend()
    plt.show()

def evaluation_metric(y_test,y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_test, y_hat)
    R2 = metrics.r2_score(y_test,y_hat)
    return MSE, RMSE, MAE, R2

def partition_dataset(sequence_length, data):
    X, y = [], []
    data_len = data.shape[0]

    for i in range(sequence_length, data_len):
        X.append(data[i - sequence_length: i, :])
        y.append(data[i, 3]) 
    
    X = np.array(X)
    y = np.array(y)
    return X, y