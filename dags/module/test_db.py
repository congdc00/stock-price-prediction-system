import psycopg2

try:
    conn = psycopg2.connect(
        host = 'localhost',
        dbname = 'postgres',
        user = 'airflow',
        password = 'airflow',
        port = 5432
    )

    conn.close()
    print("Thành công rồi")
except Exception as error:
    print(f"Lỗi lòi mắt")
    print(f" Lỗi là {error}")