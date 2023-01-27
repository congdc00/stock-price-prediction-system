import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from utils import *

data = pd.read_csv('./data/VN30_clean.csv', parse_dates=['Date'], date_parser=lambda x: datetime.strptime(x, '%Y-%m-%d'))
data.set_index('Date', inplace=True)

# Create the training and test data
idx_split = int(0.95 * len(data))
date_split = data.index[idx_split]
training_set = data.loc[: date_split, :]  # 2498
test_set = data.loc[date_split:, :]  # 130
test_set_2 = test_set.copy()

plt.figure(figsize=(10, 6))
plt.plot(training_set['Close'], label='training_set')
plt.plot(test_set['Close'], label='test_set')
plt.title('Close price')
plt.xlabel('time', fontsize=12, verticalalignment='top')
plt.ylabel('close', fontsize=14, horizontalalignment='center')
plt.legend()
plt.show()

# ARIMA
history = [x for x in training_set['Close']]
predictions = list()
for t in range(test_set.shape[0]):
    model = ARIMA(history, order=(2, 1, 0))
    model_fit = model.fit()
    yhat = model_fit.forecast()
    yhat = float(yhat[0])
    predictions.append(yhat)
    obs = test_set_2.iloc[t, 0]
    history.append(obs)

predictions1 = {
    'trade_date': test_set.index[:],
    'close': predictions
}

predictions1 = pd.DataFrame(predictions1)
predictions1 = predictions1.set_index(['trade_date'], drop=True)
plot(test_set['Close'], predictions1, title='ARIMA: Stock Price Prediction')

mse, rmse, mae, r2 = evaluation_metric(predictions1['close'], test_set['Close'])
print('MSE: %.5f' % mse)
print('RMSE: %.5f' % rmse)
print('MAE: %.5f' % mae)
print('R2: %.5f' % r2)