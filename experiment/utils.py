import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from fe import *
from datetime import datetime
from sklearn.preprocessing import RobustScaler


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

def evaluation_metric(y_test, y_hat):
    MSE = metrics.mean_squared_error(y_test, y_hat)
    RMSE = MSE**0.5
    MAE = metrics.mean_absolute_error(y_test, y_hat)
    R2 = metrics.r2_score(y_test,y_hat)
    return MSE, RMSE, MAE, R2

def partition_dataset(sequence_length, data):
    X, y = [], []
    data_len = data.shape[0]
    close_idx = FEATURES.index('Close')
    
    for i in range(sequence_length, data_len):
        X.append(data[i - sequence_length: i, :])
        y.append(data[i, close_idx])    
    X = np.array(X)
    y = np.array(y)
    return X, y

def get_final_processed_data(data_path, X_test=False):
    train_df = pd.read_csv(data_path, parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
    train_df.set_index('Date', inplace=True)

    # Get new df after feature engineering
    df_features = create_features(train_df, endog=train_df['Close'])
    data_filtered_ext = df_features[FEATURES]
    dfs = data_filtered_ext.copy()

    # Scaler 
    scaler = RobustScaler()
    np_data = scaler.fit_transform(dfs)

    if X_test:
        return np_data, data_filtered_ext, df_features 
    else:
        return np_data
