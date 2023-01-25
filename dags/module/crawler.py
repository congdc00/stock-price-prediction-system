import requests
from bs4 import BeautifulSoup
from module.option import Option
from datetime import datetime


def format_data(datetime_now, row):
    list_info = row.findAll("td", {"class": "Item_Price10"})

    close = list_info[0].text
    close = close.strip()

    volumn = list_info[2].text
    volumn = volumn.split(",")
    volumn = ".".join([volumn[0], volumn[1]])
    volumn = "{:.2f}".format(float(volumn))
    volumn = volumn +"M"

    open = list_info[5].text
    open = open.strip()

    high = list_info[6].text
    high = high.strip()
    low = list_info[7].text
    low = low.strip()

    # percent_change
    percent_change = row.find("td", {"class": "Item_ChangePrice"}).text
    percent_change = percent_change.split("(")[-1]
    percent_change = percent_change.replace(")", "")
    percent_change = percent_change.replace(" ", "")

    new_data = [datetime_now, close, open, high, low, volumn, percent_change]
    return new_data



def crawl(day_target):
    print("Starting ...")

    day_target = datetime.strptime(day_target, '%d/%m/%Y').date()

    # create the csv writer
    # title = ["Date", "Close", "Open", "High", "Low", "Volumn", "Percent Change"]

    link = "https://s.cafef.vn/Lich-su-giao-dich-VN30INDEX-1.chn"
    content = requests.get(link).content
    soup = BeautifulSoup(content, "html.parser")
    
    for i in range (0,19):

        id = Option.get_id(i)
        row = soup.find("tr", {"id": id})

        #date
        date = row.find("td", {"class": "Item_DateItem"}).text
        datetime_now = datetime.strptime(date, '%d/%m/%Y').date()
        
        if datetime_now == day_target:
            data = format_data(datetime_now, row=row)
            print(data)
            break

    print(f"Done")