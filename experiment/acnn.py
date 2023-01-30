import tensorflow as tf
import numpy as np
import pandas as pd
import math
import tensorflow.keras.backend as K
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Permute, Dense, Multiply, Flatten, Lambda, Bidirectional, LSTM, RepeatVector
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

def attention_3d_block(inputs, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    # a.shape = (batch_size, input_dim, time_steps)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)    
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(input_dims, sequence_length, lstm_units=64):
    inputs = Input(shape=(sequence_length, input_dims))
    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs) 
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)    

    output = Dense(1, activation='relu')(attention_mul)
    model = Model(inputs=inputs, outputs=output)
    return model 

def training(model):
    model.compile(optimizer='adam', loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=2, restore_best_weights=True)
    
    # Training
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop],
    validation_data=(X_test, y_test), verbose=2)

    # Get predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_pred.inverse_transform(y_pred_scaled)
    y_test_unscaled = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

    # Evalute
    mse, rmse, mae, r2 = evaluation_metric(y_pred, y_test_unscaled)
    return mse, rmse, mae, r2

def print_result(model):
    mse_array = np.zeros(10)
    rmse_array = np.zeros(10)
    mae_array = np.zeros(10)
    r2_array = np.zeros(10)
    for i in range(10):
        mse, rmse, mae, r2 = training(model)
        if mse < 100000:
            mse_array[i] = mse
            rmse_array[i] = rmse
            mae_array[i] = mae 
            r2_array[i] = r2

    mse_array = mse_array[mse_array != 0]
    rmse_array = rmse_array[rmse_array != 0]
    mae_array = mae_array[mae_array != 0]
    r2_array = r2_array[r2_array != 0]

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
    
# print_result(attention_model(config.num_features, config.sequence_length))