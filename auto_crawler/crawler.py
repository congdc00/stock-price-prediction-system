import requests
from bs4 import BeautifulSoup
from utils.option import Option
import csv
from datetime import datetime
from loguru import logger

craw4day = "17/12/2022"
name_bank = "VN30INDEX"


if __name__ == "__main__":
    logger.info("Starting ...")

    datetime_end = datetime.strptime(craw4day, '%d/%m/%Y').date()
    name_file_csv = './data/' + name_bank + '_1712_1701.csv'

    with open(name_file_csv, 'w') as f:

        # create the csv writer
        writer = csv.writer(f)
        title = ["Date", "Close", "Open", "High", "Low", "Volumn", "Percent Change"]
        writer.writerow(title)

        link = "https://s.cafef.vn/Lich-su-giao-dich-" + name_bank + "-1.chn"
        content = requests.get(link).content
        soup = BeautifulSoup(content, "html.parser")
        list_data = []
        
        for i in range (0,19):
            id = Option.get_id(i)
            row = soup.find("tr", {"id": id})

            #date
            date = row.find("td", {"class": "Item_DateItem"}).text
            datetime_now = datetime.strptime(date, '%d/%m/%Y').date()
            if datetime_now < datetime_end:
                break

            # list_info
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

            data = [datetime_now, close, open, high, low, volumn, percent_change]
            # print(datetime_now)

            # write a row to the csv file
            writer.writerow(data)

    logger.success(f"Save file in {name_file_csv}")