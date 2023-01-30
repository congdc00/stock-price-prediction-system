import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, LSTM, Dense 
from config import LSTMConfig
from fe import *
from sklearn.preprocessing import RobustScaler
from lstm import get_model
from utils import *


config = LSTMConfig()

def best_model(sequence_length, input_dims):
    inputs = Input(shape=(sequence_length, input_dims))
    lstm_layer = LSTM(200, name='lstm')(inputs)
    dense = Dense(64, activation='relu', name='dense1')(lstm_layer)
    output = Dense(1, activation='relu', name='dense2')(dense)
    model = Model(inputs=inputs, outputs=output)
    return model

def prediction_with_best_weights(df: pd.DataFrame, checkpoint_path):
    """
    Model retrain pretrained weights in entire dataset (training + test)
    Args:
        df: pd.DataFrame (Column: Close, Open, High, Low, Volumn)
        checkpoint_path: directory ('./training_pretrained_lstm/*.ckpt')
    Returns:
        next_day_close_prediction: float    
    """
    close = df['Close']

    # Add ARIMA feature
    df_features = create_features(df, endog=close)[FEATURES]

    # Scale
    np_data, data_filtered_ext, df_features = get_final_processed_data(df_features, X_test=True)
    scaler_pred = RobustScaler()
    df_close = pd.DataFrame(data_filtered_ext['Close'])
    np_close_scaled = scaler_pred.fit_transform(df_close)

    # Load best weights when training in entire data
    model = best_model(config.sequence_length, config.num_features)
    model.load_weights(checkpoint_path)
    close_prediction = scaler_pred.inverse_transform(model.predict(tf.expand_dims(np_data[-config.sequence_length: ], 0)))

    return close_prediction[0][0]

def prediction_with_pretrained_weights(df: pd.DataFrame, checkpoint_path):
    """
    Model with pretrained weights in training set
    Args:
        df: pd.DataFrame (Column: Close, Open, High, Low, Volumn)
        checkpoint_path: directory ('./training_lstm/*.ckpt')
    Returns:
        next_day_close_prediction: float    
    """
    close = df['Close']

    # Add ARIMA feature
    df_features = create_features(df, endog=close)[FEATURES]

    # Scale
    np_data, data_filtered_ext, df_features = get_final_processed_data(df_features, X_test=True)
    scaler_pred = RobustScaler()
    df_close = pd.DataFrame(data_filtered_ext['Close'])
    np_close_scaled = scaler_pred.fit_transform(df_close)

    # Load best weights when training in entire data
    model = best_model(config.sequence_length, config.num_features)
    model.load_weights(checkpoint_path)
    close_prediction = scaler_pred.inverse_transform(model.predict(tf.expand_dims(np_data[-config.sequence_length: ], 0)))

    return close_prediction[0][0]

def prediction_auto(df: pd.DataFrame, checkpoint_path):
    """
    Model retrain when has new data
    Args:
        df: pd.DataFrame (Column: Close, Open, High, Low, Volumn)
        checkpoint_path: directory ('./training_pretrained_lstm/*.ckpt')
    Returns:
        next_day_close_prediction: float    
    """
    close = df['Close']

    # Add ARIMA feature
    df_features = create_features(df, endog=close)[FEATURES]

    # Scale
    np_data, data_filtered_ext, df_features = get_final_processed_data(df_features, X_test=True)
    scaler_pred = RobustScaler()
    df_close = pd.DataFrame(data_filtered_ext['Close'])
    np_close_scaled = scaler_pred.fit_transform(df_close)

    # Partition dataset
    X, y = partition_dataset(np_data, config=config.sequence_length)

    # Load best weights when training in entire data
    model = best_model(config.sequence_length, config.num_features)
    model.load_weights(checkpoint_path)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, batch_size=config.batch_size, epochs=10, verbose=0)
    close_prediction = scaler_pred.inverse_transform(model.predict(tf.expand_dims(np_data[-config.sequence_length: ], 0)))

    return close_prediction[0][0]