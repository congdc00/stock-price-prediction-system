import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone, date, timedelta

class Option():
  
    @staticmethod
    def get_id(i: int) -> str:
        if i % 2 == 0:
            id = "ContentPlaceHolder1_ctl03_rptData2" + "_itemTR_" + str(i)
        else:
            id = "ContentPlaceHolder1_ctl03_rptData2" + "_altitemTR_" + str(i)
        return id

def format_data(datetime_now, row):
    datetime_now = datetime_now.strftime('%Y-%m-%d')
    list_info = row.findAll("td", {"class": "Item_Price10"})

    close = list_info[0].text
    close = close.strip()
    close = close.replace(",", "")

    volumn = list_info[2].text
    volumn = volumn.split(",")
    volumn = ".".join([volumn[0], volumn[1]])
    volumn = "{:.2f}".format(float(volumn))

    open = list_info[4].text
    open = open.strip()
    open = open.replace(",", "")

    high = list_info[5].text
    high = high.strip()
    high = high.replace(",", "")

    low = list_info[6].text
    low = low.strip()
    low = low.replace(",", "")

    # percent_change
    percent_change = row.find("td", {"class": "Item_ChangePrice"}).text
    percent_change = percent_change.split("(")[-1]
    percent_change = percent_change.replace(")", "")
    percent_change = percent_change.replace(" ", "")
    percent_change = percent_change.replace("%", "")

    new_data = {'datetime_now': datetime_now, 'close': close, 'open': open, 'high': high, 'low': low, 'volumn': volumn, 'percent_change': percent_change}
    # Convert the date to a datetime object
    dt_object = datetime.strptime(datetime_now, '%Y-%m-%d')

    # Get the timestamp
    timestamp = dt_object.timestamp()
    new_data['timestamp'] = timestamp
    return new_data

def crawlStock(symbol='VN30INDEX'):
    print("Starting crawl: {}".format(symbol))
    latest_execution_date = date.today()
    # day_target = datetime.strptime( "27/1/2023", '%d/%m/%Y').date()
    day_target = latest_execution_date - timedelta(days=30)
    print(f"Ngày hiện tại-------- {latest_execution_date} va {day_target}")

    # create the csv writer
    # title = ["Date", "Close", "Open", "High", "Low", "Volumn", "Percent Change"]

    link = "https://s.cafef.vn/Lich-su-giao-dich-{}-1.chn".format(symbol)
    
    content = requests.get(link).content
    soup = BeautifulSoup(content, "html.parser")
    
    stockData = []
    for i in range (0,20):

        id = Option.get_id(i)
        row = soup.find("tr", {"id": id})

        #date
        date_data = row.find("td", {"class": "Item_DateItem"}).text
        datetime_now = datetime.strptime(date_data, '%d/%m/%Y').date()
        
        date_data = format_data(datetime_now, row=row)
        date_data['symbol'] = symbol
        # if datetime_now == day_target:
        #     date_data = format_data(datetime_now, row=row)
        #     print(date_data)
        #     break
        stockData.append(date_data)
    
    if stockData!=[]:
        print(stockData)
        print(len(stockData))
    print(f"Done")
    
    return stockData