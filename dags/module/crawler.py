import requests
from bs4 import BeautifulSoup
from module.option import Option
import psycopg2
from datetime import datetime, timezone, date


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

    print(f"Dữ liệu crăl về là :{list_info}")
    low = list_info[6].text
    low = low.strip()
    low = low.replace(",", "")

    # percent_change
    percent_change = row.find("td", {"class": "Item_ChangePrice"}).text
    percent_change = percent_change.split("(")[-1]
    percent_change = percent_change.replace(")", "")
    percent_change = percent_change.replace(" ", "")
    percent_change = percent_change.replace("%", "")

    new_data = [datetime_now, close, open, high, low, volumn, percent_change]
    return new_data

#database
def update_db(data):
    try:
        conn = psycopg2.connect(
            host = 'host.docker.internal',
            dbname = 'postgres',
            user = 'airflow',
            password = 'airflow',
            port = 5432
        )
        cur = conn.cursor()
        create_script = ''' CREATE TABLE IF NOT EXISTS stock (
                                datetime_now    varchar(40) PRIMARY KEY,
                                close   float(2),
                                open    float(2),
                                high    float(2),
                                low float(2),
                                volumn  float(2),
                                percent_change  float(2))'''
        cur.execute(create_script)

        insert_script = 'INSERT INTO stock (datetime_now, close, open, high, low, volumn, percent_change) VALUES (%s,%s,%s,%s,%s,%s,%s)'
        insert_value = tuple(data)
        print(f"Thêm data {insert_value}")
        cur.execute(insert_script, insert_value)
        conn.commit()
        conn.close()
        print("Thành công rồi")
    except Exception as error:
        print(f"Lỗi lòi mắt")
        print(f" Lỗi là {error}")

    

def crawl():
    print("Starting ...")
    latest_execution_date = date.today()
    day_target = datetime.strptime( "27/1/2023", '%d/%m/%Y').date()
    print(f"Ngày hiện tại-------- {latest_execution_date} va {day_target}")
    

    # create the csv writer
    # title = ["Date", "Close", "Open", "High", "Low", "Volumn", "Percent Change"]

    link = "https://s.cafef.vn/Lich-su-giao-dich-VN30INDEX-1.chn"
    content = requests.get(link).content
    soup = BeautifulSoup(content, "html.parser")
    date_data = []
    for i in range (0, 19):

        id = Option.get_id(i)
        row = soup.find("tr", {"id": id})

        #date
        date_data = row.find("td", {"class": "Item_DateItem"}).text
        datetime_now = datetime.strptime(date_data, '%d/%m/%Y').date()
        
        if datetime_now == day_target:
            date_data = format_data(datetime_now, row=row)
            print(date_data)
            break
    if date_data!=[]:
        update_db(date_data)
    print(f"Done")