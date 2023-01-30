import tensorflow as tf
import numpy as np
import pandas as pd
import math
from datetime import datetime
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.preprocessing import RobustScaler
from utils import *
from fe import *
from config import LSTMConfig

# config
config = LSTMConfig()
SEQUENCE_LENGTH = config.sequence_length 
EPOCHS = config.epochs
BATCH_SIZE = config.batch_size
PATIENCE = config.patience

train_df = pd.read_csv('./data/VN30_clean.csv', parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
train_df.set_index('Date', inplace=True)
np_data, data_filtered_ext, df_features = get_final_processed_data(train_df, X_test=True)

# Creating a separate scaler that works on a single column for scaling predictions
scaler_pred = RobustScaler()
df_close = pd.DataFrame(data_filtered_ext['Close'])
np_close_scaled = scaler_pred.fit_transform(df_close)

train_data_len, (X_train, y_train), (X_test, y_test) = get_X_y(SEQUENCE_LENGTH, np_data)
# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)

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

def training(model_type, save_weights=False, **kwargs):
    model = get_model(model_type)
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=0, restore_best_weights=True)
    
    # Training
    if save_weights:
        cp_callback = ModelCheckpoint(filepath='training_lstm/cp_{}.ckpt'.format(kwargs['checkpoint']),
                                    monitor='loss',
                                    save_weights_only=True,
                                    save_best_only=True,
                                    verbose=0)
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop, cp_callback],
                validation_data=(X_test, y_test), verbose=2)
    else:
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop],
        validation_data=(X_test, y_test), verbose=2)

    # Get predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

    # Evalute
    mse, rmse, mae, r2 = evaluation_metric(y_pred, y_test_unscaled)
    return mse, rmse, mae, r2

# Evaluate
def print_result(model_type):
    mse_array = np.zeros(10)
    rmse_array = np.zeros(10)
    mae_array = np.zeros(10)
    r2_array = np.zeros(10)
    for i in range(10):
        mse, rmse, mae, r2 = training(model_type, checkpoint=i)
        mse_array[i] = mse
        rmse_array[i] = rmse
        mae_array[i] = mae 
        r2_array[i] = r2

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

# single-layer LSTM is not only the one with the best time efficiency but also the one with the best performance

def get_predict_by_pretrained(checkpoint_path):
    """
    Args:
        checkpoint_path: directory
    Returns
        predict_df: pd.DataFrame
    """
    model = get_model(model_type=1)
    model.load_weights(checkpoint_path)
    y_pred_scaled = model.predict(X_test)
    # Unscale the predicted values
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))
    predict_df = pd.DataFrame({"Price": y_test_unscaled.reshape(-1),
                           "Predict": y_pred.reshape(-1)})
    predict_df.index = df_features[df_features.index[train_data_len]: ].index
    return predict_df

# checkpoint_1 has best performance
# predict_df = get_predict_by_pretrained('training_lstm/cp_1.ckpt')
# plot(predict_df['Price'], predict_df['Predict'], 'LSTM: Stock Price Prediction')
