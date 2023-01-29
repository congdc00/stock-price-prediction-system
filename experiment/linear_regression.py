import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from utils import *
from sklearn.linear_model import LinearRegression, Ridge


# Sliding window
def trading_window(data, n):
    data['Target'] = data['Close'].shift(-n)
    return data

data = pd.read_csv('./data/VN30_clean.csv', parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
data.set_index('Date', inplace=True)
data = trading_window(data, 1)
data = data.dropna()
data_2 = data.copy()

# Scaler
scaler = RobustScaler()
training_scaled = scaler.fit_transform(data)

# Seperate scaler that works on a single column for scaling predictions
scaler_pred = RobustScaler()
pred_scaled = scaler_pred.fit_transform(data_2['Close'].values.reshape(-1, 1))

# Create the training and test data
training_data_len = int(len(data) *  0.95)
X_train, y_train = training_scaled[:training_data_len, :5], training_scaled[:training_data_len, -1]
X_test, y_test = training_scaled[training_data_len:, :5], training_scaled[training_data_len:, -1]

# Linear Regression 
def regression(ridge=False):
    linear = Ridge(alpha=1 if ridge else 0)
    return linear

linear = regression(ridge=False)
linear.fit(X_train, y_train)

y_predict = linear.predict(X_test)
y_predict = scaler_pred.inverse_transform(y_predict.reshape(-1, 1))
y_test = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

predict_df = pd.DataFrame({"Price": y_test.reshape(-1),
                           "Predict": y_predict.reshape(-1)})

predict_df.index = data[data.index[training_data_len]: ].index

# Plot
plot(predict_df['Price'], predict_df['Predict'], title='Linear Regression: Stock Price Prediction')

mse, rmse, mae, r2 = evaluation_metric(y_predict, y_test)
print(f'MSE: {mse:.5f}')
print(f'RMSE: {rmse:.5f}')
print(f'MAE: {mae:.5f}')
print(f'R2: {r2:.5f}')
