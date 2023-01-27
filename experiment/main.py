import tensorflow as tf
import numpy as np
import pandas as pd
import math
from datetime import datetime
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dense, Dropout 
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import RobustScaler
from utils import *
from fe import *


# config
class Config:
    def __init__(self):
        self.sequence_length = 15
        self.epochs = 100 
        self.batch_size = 32
        self.PATIENCE = 8

config = Config()
SEQUENCE_LENGTH = config.sequence_length 
EPOCHS = config.epochs
BATCH_SIZE = config.batch_size
PATIENCE = config.PATIENCE

train_df = pd.read_csv('./data/VN30_clean.csv', parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
train_df.set_index('Date', inplace=True)

# Get new df after feature engineering
df_features = create_features(train_df, endog=train_df['Close'])
data_filtered_ext = df_features[FEATURES]
dfs = data_filtered_ext.copy() 

# Scaler
scaler = RobustScaler()
np_data = scaler.fit_transform(dfs) 

# Generate partition data
X, y = partition_dataset(SEQUENCE_LENGTH, np_data)

if __name__ == '__main__':
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    lstm_layer = LSTM(200)(inputs)
    dense = Dense(64, activation='relu')(lstm_layer)
    output = Dense(1, activation='relu')(dense)
    model = Model(inputs=inputs, outputs=output)

    cp_callback = ModelCheckpoint(filepath='training_lstm/cp.ckpt',
                                monitor='loss',
                                save_weights_only=True,
                                save_best_only=True,
                                verbose=0)
    early_stop = EarlyStopping(monitor='loss', patience=PATIENCE, verbose=1, restore_best_weights=True)
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stop, cp_callback], verbose=2)