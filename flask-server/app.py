import os
from flask import Flask, redirect, url_for, request, render_template, jsonify, request
from pymongo import MongoClient
from crawler.crawler import crawlStock

app = Flask("StockPrediction")

# To change accordingly 
# print(os.environ)
# client = MongoClient(os.environ["DB_PORT_27017_TCP_ADDR"], 27017)
client = MongoClient('mongodb://localhost:27017/')
db = client.stock_prediction

# def _crawlStock():
    

@app.route("/")
def index():
    stockItems = db.stocks.find() or []
    print(stockItems)
    stockSymbols = [stockItem['symbol'] for stockItem in stockItems]

    return render_template("index.html", stockSymbols=stockSymbols)

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
    
    # save data to db
    db.stock_history.insert_many(stockData)
    
    # get data from db    
    return jsonify({"message": "Stocks inserted successfully"}), 201

@app.route("/api/getStockPrediction", methods=["POST"])
def getStockPrediction():
    return {  }

@app.route("/new", methods=["POST"])
def new():
    data = {
        "helloworld": request.form["helloworld"]
    }

    db.appdb.insert_one(data)

    return redirect(url_for("index"))
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)