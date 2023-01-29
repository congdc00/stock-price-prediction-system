import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# Feature Engineering
# List of considered Features
# List of considered Features
FEATURES = [
             'Close',
             'Open',
             'High',
             'Low',
             'ARIMA_resid',
#             'Volumn',
#             'Day',
#             'Month',
#             'Year',
#             'close_shift-1',
#             'close_shift-2',
#             'MACD',
#             'RSI',
#             'MA200',
#             'MA200_high',
#             'MA200_low',
#             'Bollinger_Upper',
#             'Bollinger_Lower',
#             'MA100',            
#             'MA50',
#             'MA26',
#             'MA14_low',
#             'MA14_high',
#             'MA12',
#             'EMA20',
#             'EMA100',
#             'EMA200',
#             'DIFF-MA200-MA50',
#             'DIFF-MA200-MA100',
#             'DIFF-MA200-CLOSE',
#             'DIFF-MA100-CLOSE',
#             'DIFF-MA50-CLOSE'
           ]

def arima_features(endog):
    """
    Args:
        endog : array_like
    Returns
        arima_resid: array_like
    """
    arima_model = ARIMA(endog=endog, order=(2, 1, 0)).fit()
    arima_resid = arima_model.resid.values
    arima_resid[0] = 0
    return arima_resid

# Feature Engineering
def create_features(df, arima=True if 'ARIMA_resid' in FEATURES else False, endog=None):
    """
    Args:
        df: Dataframe
        endog: array_like if FEATURES has ARIMA_resid else default (None)
    Returns:
        df_features: Dataframe
    """
    
    df = pd.DataFrame(df)
    # ARIMA residual
    if arima:
        df['ARIMA_resid'] = arima_features(endog)

    # Day, Month, Year
    d = pd.to_datetime(df.index)
    df['Day'] = d.strftime("%d") 
    df['Month'] = d.strftime("%m") 
    df['Year'] = d.strftime("%Y") 
    
    # Moving averages - different periods
    df['MA200'] = df['Close'].rolling(window=200).mean() 
    df['MA100'] = df['Close'].rolling(window=100).mean() 
    df['MA50'] = df['Close'].rolling(window=50).mean() 
    df['MA26'] = df['Close'].rolling(window=26).mean() 
    df['MA20'] = df['Close'].rolling(window=20).mean() 
    df['MA12'] = df['Close'].rolling(window=12).mean() 
    
    # SMA Differences - different periods
    df['DIFF-MA200-MA50'] = df['MA200'] - df['MA50']
    df['DIFF-MA200-MA100'] = df['MA200'] - df['MA100']
    df['DIFF-MA200-CLOSE'] = df['MA200'] - df['Close']
    df['DIFF-MA100-CLOSE'] = df['MA100'] - df['Close']
    df['DIFF-MA50-CLOSE'] = df['MA50'] - df['Close']
    
    # Moving Averages on high, lows, and std - different periods
    df['MA200_low'] = df['Low'].rolling(window=200).min()
    df['MA14_low'] = df['Low'].rolling(window=14).min()
    df['MA200_high'] = df['High'].rolling(window=200).max()
    df['MA14_high'] = df['High'].rolling(window=14).max()
    df['MA20dSTD'] = df['Close'].rolling(window=20).std() 
    
    # Exponential Moving Averages (EMAS) - different periods
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    # Shifts (one day before and two days before)
    df['close_shift-1'] = df.shift(1)['Close']
    df['close_shift-2'] = df.shift(2)['Close']

    # Bollinger Bands
    df['Bollinger_Upper'] = df['MA20'] + (df['MA20dSTD'] * 2)
    df['Bollinger_Lower'] = df['MA20'] - (df['MA20dSTD'] * 2)
    
    # Relative Strength Index (RSI)
    df['K-ratio'] = 100*((df['Close'] - df['MA14_low']) / (df['MA14_high'] - df['MA14_low']) )
    df['RSI'] = df['K-ratio'].rolling(window=3).mean() 

    # Moving Average Convergence/Divergence (MACD)
    df['MACD'] = df['EMA12'] - df['EMA26']
    
    # Replace nan
    nareplace = df.at[df.index.max(), 'Close']    
    df.fillna((nareplace), inplace=True)
    
    return df