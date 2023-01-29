import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, LSTM, Dense 
from models.config import LSTMConfig
from models.fe import *
from sklearn.preprocessing import RobustScaler


config = LSTMConfig()

def prediction(df: pd.DataFrame, checkpoint_path):
  """
  Args:
      df: DataFrame (Column: Close, Open, High, Low, Volumn)
      checkpoint_path: directory
  Returns:
      next_day_close_prediction: float
  """
  close = df['Close']

  df_features = create_features(df, endog=close)[FEATURES]
  df_features_copy = df_features.copy()
  scaler = RobustScaler()
  np_data = scaler.fit_transform(df_features_copy)

  scaler_pred = RobustScaler()
  df_close = pd.DataFrame(df_features_copy['Close'])
  np_close_scaled = scaler_pred.fit_transform(df_close)

  inputs = Input(shape=(config.sequence_length, config.num_features))
  lstm_layer = LSTM(200, name='lstm')(inputs)
  dense = Dense(64, activation='relu', name='dense1')(lstm_layer)
  output = Dense(1, activation='relu', name='dense2')(dense)
  model = Model(inputs=inputs, outputs=output)
  model.load_weights(checkpoint_path)
  close_prediction = scaler_pred.inverse_transform(model.predict(tf.expand_dims(np_data[-config.sequence_length: ], 0)))

  return close_prediction[0][0]