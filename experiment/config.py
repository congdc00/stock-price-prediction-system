class BaseConfig():
    batch_size = 32
    sequence_length = 15
    patience = 8
    epochs = 100
    data_path = './data/VN30_clean.csv'

class LSTMConfig(BaseConfig):
    lstm_unit = 200