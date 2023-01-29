import os
from flask import Flask, redirect, url_for, request, render_template, jsonify, request
from bson.json_util import dumps
from pymongo import MongoClient, ASCENDING, InsertOne
import json
from datetime import datetime
from crawler.crawler import crawlStock
import csv
import plotly.express as px
import plotly
import pandas as pd
from models.predict import prediction

app = Flask("StockPrediction")

# To change accordingly 
# print(os.environ)
# client = MongoClient(os.environ["DB_PORT_27017_TCP_ADDR"], 27017)
client = MongoClient('mongodb://localhost:27017/')
db = client.stock_prediction

def getStockPredictionPrice():
    # Fetch stock data
    stockDataCursor = db.stock_history.find({'symbol': 'VN30INDEX'},{"_id":0}).sort('timestamp', 1)
        
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
    
    df = pd.DataFrame(stock_dict)
    checkpoint_path = 'models/checkpoint/cp.ckpt'
    
    result = prediction(df, checkpoint_path)
    
    return result

@app.route("/")
def index():
    stockItems = db.stocks.find() or []
    stockSymbols = [stockItem['symbol'] for stockItem in stockItems]
    # Fetch stock data
    stockDataCursor = db.stock_history.find({'symbol': 'VN30INDEX'},{"_id":0}).sort('timestamp', -1).limit(100)
    
    # list_cur = list(stockDataCursor)
    # json_data = dumps(list_cur)
    # stock_df = DataFrame(json_data)
    stock_dict = {}
    stock_dict['Date'] = []
    stock_dict['Close Price'] = []
    for stockData in stockDataCursor:
        stock_dict['Close Price'].append(float(stockData['close']))
        stock_dict['Date'].append(stockData['datetime_now'])
    
    
    df = pd.DataFrame(stock_dict)
    fig = px.line(df, x='Date', y='Close Price')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    predictedClosePrice = getStockPredictionPrice()
    latestClosedPrice = stock_dict['Close Price'][0]
    print(latestClosedPrice)
    print(predictedClosePrice)
    
    return render_template("index.html", stockSymbols=stockSymbols, graphJSON=graphJSON, predictedClosePrice=predictedClosePrice, latestClosedPrice=latestClosedPrice)

# add new stocks
@app.route("/api/addStocks", methods=["POST"])
def addStocks():
    # Get the list of stocks from the request body
    stocks = request.get_json()['stocks']

    # Insert the stocks into the collection
    db.stocks.insert_many(stocks)

    return jsonify({"message": "Stocks inserted successfully"}), 201

@app.route("/api/getStockAnalysis", methods=["POST"])
def getStockAnalysis():
    stockSymbol = request.get_json()['symbol']
    
    # crawl data
    stockData = crawlStock(stockSymbol)
    
    # check first time insert
    if 'stock_history' not in db.list_collection_names():
        db.stock_history.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING)], unique=True)
    
    lastestStockHistory = db.stock_history.find_one({'symbol': stockSymbol}, sort=[("timestamp", -1)]) or {}
    
    # save data to db
    if len(stockData) != 0:
        lastestTimestamp = 0 if not lastestStockHistory else lastestStockHistory["timestamp"]
        
        operations = [InsertOne(doc) for doc in stockData if doc["timestamp"] > lastestTimestamp]
        if len(operations) != 0:
            result = db.stock_history.bulk_write(operations, ordered=False)
    # get data from db
    stockDataCursor = db.stock_history.find({'symbol': stockSymbol}).sort('timestamp', 1)

    results = list(stockDataCursor)
    json_data = dumps(results)
    return json_data, 200, {'Content-Type': 'application/json'}

@app.route("/api/getStockPrediction", methods=["POST"])
def getStockPrediction():
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
    df = df[:-1]
    print(df)
    checkpoint_path = 'models/checkpoint/cp.ckpt'
    
    result = prediction(df, checkpoint_path)
    
    return jsonify({"message": "Stocks inserted successfully"}), 201

@app.route("/api/addDataToDBFromCSV", methods=["POST"])
def addDataToDBFromCSV():
    # check first time insert
    if 'stock_history' not in db.list_collection_names():
        db.stock_history.create_index([("symbol", ASCENDING), ("timestamp", ASCENDING)], unique=True)
    with open('crawler/data/VN30_clean.csv', newline='', encoding='utf-8') as csvfile:
        stock_reader = csv.DictReader(csvfile)
        convertedData = []
        for row in stock_reader:
            # stock_collection.insert_one(row)
            stockDoc = {
                "datetime_now": row["Date"],
                "close": row["Close"],
                "open": row["Open"],
                "high": row["High"],
                "low": row["Low"],
                "symbol": "VN30INDEX",
                "volumn": row["Volumn"],
            }
            
            # Convert the date to a datetime object
            dt_object = datetime.strptime(stockDoc["datetime_now"], '%Y-%m-%d')

            # Get the timestamp
            timestamp = dt_object.timestamp()
            stockDoc['timestamp'] = timestamp
            convertedData.append(stockDoc)
    lastestStockHistory = db.stock_history.find_one(sort=[("timestamp", -1)]) or {}
    
    # save data to db
    if len(convertedData) != 0:
        lastestTimestamp = 0 if not lastestStockHistory else lastestStockHistory["timestamp"]
        
        operations = [InsertOne(doc) for doc in convertedData if doc["timestamp"] > lastestTimestamp]
        if len(operations) != 0:
            result = db.stock_history.bulk_write(operations, ordered=False)
    return jsonify({"message": "Stocks inserted successfully"}), 201
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)