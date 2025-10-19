"""
Simple Amazon Data Loading DAG
"""

import os
from datetime import timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

# Simple DAG configuration
dag = DAG(
    'amazon_data_loader',
    default_args={
        'owner': 'nourhan-elsehrawy',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='Load Amazon data into PostgreSQL',
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['amazon-electronics', 'postgresql', 'data-loading']
)

# Load data to PostgreSQL
load_data_task = BashOperator(
    task_id='load_amazon_data',
    bash_command='cd /opt/airflow && python utils/db_process_amazon_data.py',
    dag=dag
)

