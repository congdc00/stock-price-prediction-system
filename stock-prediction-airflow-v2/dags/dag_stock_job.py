from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.mongo.hooks.mongo import MongoHook
from datetime import datetime,timedelta
from module.crawler import crawlStock
from module.prediction import prediction
import pandas as pd

def crawl():
  try:
    hook = MongoHook(mongo_conn_id='mongoid')
    client = hook.get_conn()
    db = client.stock_prediction
    print(f"Connected to MongoDB - {client.server_info()}")
    
    stockData = crawlStock()
    lastestStockHistory = db.stock_history.find_one({'symbol': 'VN30INDEX'}, sort=[("timestamp", -1)]) or {}
    
    # save data to db
    if len(stockData) != 0:
      lastestTimestamp = 0 if not lastestStockHistory else lastestStockHistory["timestamp"]
      
      operations = [doc for doc in stockData if doc["timestamp"] > lastestTimestamp]
      if len(operations) != 0:
        result = db.stock_history.insert_many(operations, ordered=False)
    
  except Exception as e:
    print("Error connecting to MongoDB -- {}".format(e))

def predict():
  try:
    hook = MongoHook(mongo_conn_id='mongoid')
    client = hook.get_conn()
    db = client.stock_prediction
    print(f"Connected to MongoDB - {client.server_info()}")
    
    # Fetch stock data
    stockDataCursor = db.stock_history.find({'symbol': 'VN30INDEX'},{"_id":0}).sort('timestamp', 1)
    
    # list_cur = list(stockDataCursor)
    # json_data = dumps(list_cur)
    # stock_df = DataFrame(json_data)
    print(stockDataCursor)
    
        
    stock_dict = {}
    stock_dict['Close'] = []
    stock_dict['Open'] = []
    stock_dict['High'] = []
    stock_dict['Low'] = []
    stock_dict['Volumn'] = []
    
    for stockData in stockDataCursor:
        stock_dict['Close'].append(float(stockData['close']))
        stock_dict['Open'].append(stockData['open'])
        stock_dict['High'].append(stockData['high'])
        stock_dict['Low'].append(stockData['low'])
        stock_dict['Volumn'].append(stockData['volumn'])
        
    
    print(len(stock_dict['Close']))
    print(len(stock_dict['Open']))
    print(len(stock_dict['High']))
    print(len(stock_dict['Low']))
    print(len(stock_dict['Volumn']))
    
    df = pd.DataFrame(stock_dict)
    print(df)
    checkpoint_path = 'dags/module/checkpoint/cp.ckpt'
    result = prediction(df, checkpoint_path)
    print(result)
    
    lastestStockHistory = db.stock_history.find_one({'symbol': 'VN30INDEX'}, sort=[("timestamp", -1)]) or {}
    
    db.stock_prediction.insert_one({
      'symbol': 'VN30INDEX',
      'lastDate': lastestStockHistory['datetime_now'],
      'lastTimestamp': lastestStockHistory['timestamp'],
      'predictedClosePrice': round(float(result), 2),
    })
    
  except Exception as e:
    print("Error connecting to MongoDB -- {}".format(e))

default_args = {
  'owner': 'CongDC',
  'retries': 5,
  'retry_delay': timedelta(minutes=5)
}

with DAG(
  default_args = default_args,
  dag_id='auto_stock',
  description='craw data from cafeF',
  start_date = datetime(2023, 1, 26),
  schedule_interval = '@daily',
) as dag:
  task1 = PythonOperator(
    task_id = 'task_id_1',
    python_callable=crawl
  )
  
  task2 = PythonOperator(
    task_id = 'task_id_2',
    python_callable=predict
  )

  task1 >> task2