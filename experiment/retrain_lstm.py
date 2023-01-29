import tensorflow as tf
import numpy as np
import pandas as pd
import os
from datetime import datetime
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, LSTM, Dense 
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
LSTM_UNIT = config.lstm_unit
DATA_PATH = config.data_path

np_data = get_final_processed_data(DATA_PATH)

# Generate partition data
X, y = partition_dataset(SEQUENCE_LENGTH, np_data)

# Get model
def get_pretrained_model(checkpoint_path):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    lstm_layer = LSTM(200, name='lstm')(inputs)
    dense = Dense(64, activation='relu', name='dense1')(lstm_layer)
    output = Dense(1, activation='relu', name='dense2')(dense)
    model = Model(inputs=inputs, outputs=output)
    model.load_weights(checkpoint_path)
    return model

# retrain in entire dataset with pretrained weights
if not os.path.exists('training_pretrained_lstm/cp.ckpt.index'):
    cp_callback = ModelCheckpoint(
        filepath='training_pretrained_lstm/cp.ckpt',
        monitor='loss',
        save_weights_only=True,
        save_best_only=True,
        verbose=0
    )
    early_stop = EarlyStopping(
        monitor='loss', 
        patience=PATIENCE, 
        verbose=0, 
        restore_best_weights=True
    )
    # checkpoint_1 has best performance
    pretrained_model = get_pretrained_model('training_lstm/cp_1.ckpt')
    # unfreeze all layers
    for layer in pretrained_model.layers:
        layer.trainable = True
    pretrained_model.compile(optimizer='adam', loss='mse')
    pretrained_model.fit(X, y, batch_size=BATCH_SIZE, epochs=20, callbacks=[early_stop, cp_callback], verbose=2)