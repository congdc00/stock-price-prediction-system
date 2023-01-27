import tensorflow as tf
import numpy as np
import pandas as pd
import math
from datetime import datetime
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.preprocessing import RobustScaler
from utils import *
from fe import *

SEQUENCE_LENGTH = 15 
EPOCHS = 100 
BATCH_SIZE = 32
PATIENCE = 8

train_df = pd.read_csv('./data/VN30_clean.csv', parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
train_df.set_index('Date', inplace=True)

# Get new df after feature engineering
df_features = create_features(train_df, endog=train_df['Close'])
data_filtered_ext = df_features[FEATURES]
dfs = data_filtered_ext.copy() 

# Scaler
scaler = RobustScaler()
np_data = scaler.fit_transform(dfs) 

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = RobustScaler()
df_close = pd.DataFrame(data_filtered_ext['Close'])
np_close_scaled = scaler_pred.fit_transform(df_close)

# Create the training and test data
train_data_len = math.ceil(np_data.shape[0] * 0.95)

# Create the training and test data
train_data = np_data[:train_data_len, :]
test_data = np_data[train_data_len - SEQUENCE_LENGTH:, :]

# Generate training data and test data
X_train, y_train = partition_dataset(SEQUENCE_LENGTH, train_data)
X_test, y_test = partition_dataset(SEQUENCE_LENGTH, test_data)

# print(X_train.shape, y_train.shape)
# print(X_train.shape, y_test.shape)

def get_model(model_type):
    """
    model type：
        1. single-layer LSTM
        2. multi-layer LSTM
        3. bidirectional LSTM
    """
    model = Sequential()
    if model_type == 1:
        model.add(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='relu'))
    elif model_type == 2:
        model.add(LSTM(200, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) 
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='relu'))
    elif model_type == 3:
        model.add(Bidirectional(LSTM(200, input_shape=(X_train.shape[1], X_train.shape[2]))))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='relu'))
    return model

def training(model_type):
    model = get_model(model_type)
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, restore_best_weights=True)
    history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop],
                      validation_data=(X_test, y_test), verbose=0)
    return model

def get_predict(model_type):
    model = training(model_type)
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    return y_pred, y_test_unscaled

def evaluate(model_type):
    y_pred, y_test_unscaled = get_predict(model_type)
    mse, rmse, mae, r2 = evaluation_metric(y_pred, y_test_unscaled)
    return mse, rmse, mae, r2

# Evaluate
# single-layer LSTM is not only the one with the best time efficiency but also the one with the best performance
model_type = 1
mse_array = np.zeros(10)
rmse_array = np.zeros(10)
mae_array = np.zeros(10)
r2_array = np.zeros(10)
for i in range(10):
    mse, rmse, mae, r2 = evaluate(model_type)
    mse_array[i] = mse
    rmse_array[i] = rmse
    mae_array[i] = mae
    r2_array[i] = r2
    print(mse_array)

mean_mse = np.mean(mse_array)
std_mse = np.std(mse_array)
print(f'MSE: {mean_mse:.5f} +- {std_mse:.5f}')

mean_rmse = np.mean(rmse_array)
std_rmse = np.std(rmse_array)
print(f'RMSE: {mean_rmse:.5f} +- {std_rmse:.5f}')

mean_mae = np.mean(mae_array)
std_mae = np.std(mae_array)
print(f'MAE: {mean_mae:.5f} +- {std_mae:.5f}')

mean_r2 = np.mean(r2_array)
std_r2 = np.std(r2_array)
print(f'R2: {mean_r2:.5f} +- {std_r2:.5f}')
