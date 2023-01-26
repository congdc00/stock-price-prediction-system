import pandas as pd

VN30_df = pd.read_csv('./data/VN30_full.csv')

def obj2float(x):
    return float(x.replace(',', ''))

VN30_df.dropna(inplace=True)
VN30_df['Date'] = pd.to_datetime(VN30_df['Date'], format='%m/%d/%Y')
VN30_df['Close'] = VN30_df['Close'].apply(obj2float)
VN30_df['Open'] = VN30_df['Open'].apply(obj2float)
VN30_df['High'] = VN30_df['High'].apply(obj2float)
VN30_df['Low'] = VN30_df['Low'].apply(obj2float)
VN30_df['Volumn'] = VN30_df['Volumn'].apply(lambda x: 1e6 * float(x.replace('M', '')))
VN30_df.drop('Percent Change', axis=1, inplace=True)    
VN30_df = VN30_df[::-1]

VN30_df.to_csv('./data/VN30_clean.csv', index=False)