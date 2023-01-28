from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from module.crawler import crawl
from airflow.providers.postgres.operators.postgres import PostgresOperator

default_args = {
    'owner': 'CongDC',
    'retries': 5,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    default_args = default_args,
    dag_id='auto_craw81',
    description='craw data from cafeF',
    start_date = datetime(2023, 1, 26),
    schedule_interval = '@daily',
) as dag:
    task1 = PythonOperator(
        task_id = 'task_id_1',
        python_callable=crawl
    )
    task1