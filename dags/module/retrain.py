import requests
from bs4 import BeautifulSoup
from option import Option
import psycopg2
from datetime import datetime, timezone, date

def retrain():
    try:
        conn = psycopg2.connect(
            host = 'host.docker.internal',
            dbname = 'postgres',
            user = 'airflow',
            password = 'airflow',
            port = 5432
        )
        cur = conn.cursor()
        
        cur.execute("select * from stock")
        record = cur.fetchall()
        print(record)
        conn.commit()
        conn.close()
        print("Thành công rồi")
    except Exception as error:
        print(f"Lỗi lòi mắt")
        print(f" Lỗi là {error}")

retrain()