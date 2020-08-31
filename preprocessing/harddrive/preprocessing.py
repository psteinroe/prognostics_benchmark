import os
import sys
import math
import pandas as pd
import logging
from pymongo import MongoClient
from multiprocessing import Pool
import tqdm
import time

sys.path.append('../../')
from prognostics_benchmark.datasets.manager import DataManager

logging.basicConfig(level=logging.NOTSET)


def load_parallel():
    input_path = os.path.join('raw')
    datasets = [dataset.path for dataset in os.scandir(input_path) if dataset.is_dir()]
    p = Pool(6)
    for _ in tqdm.tqdm(p.imap_unordered(load_job, datasets), total=len(list(datasets))):
        pass


def load_job(dataset):
    client = MongoClient(CON_STRING)
    db = client.harddrive_preprocessing

    buffer = {}

    for datafile in sorted(os.scandir(dataset), key=lambda e: e.name):
        if datafile.name.endswith('.csv') is False:
            continue

        df_in = pd.read_csv(datafile.path, header=0)

        for model, df_model in df_in.groupby('model'):
            if model not in buffer:
                buffer[model] = df_model
            else:
                buffer[model] = pd.concat([buffer[model], df_model], ignore_index=True)

    for model, df in buffer.items():
        records = [{k: v for k, v in item.items() if type(v) is str or math.isnan(v) is False} for item in
                   df.to_dict('records')]
        if len(records) > 0:
            db[model].insert_many(records)

    buffer.clear()


def create_index():
    client = MongoClient("mongodb://localhost:27017/")
    db_preprocessing = client.harddrive_preprocessing

    for idx, model in enumerate(db_preprocessing.list_collection_names()):
        print(str(idx) + '/' + str(len(db_preprocessing.list_collection_names()) - 1))
        db_preprocessing[model].create_index('serial_number', background=True)


def prepare_parallel():
    local_client = MongoClient(CON_STRING)
    db_preprocessing = local_client.harddrive_preprocessing
    collection_names = db_preprocessing.list_collection_names()

    p = Pool(6)
    for _ in tqdm.tqdm(p.imap_unordered(prepare_job, collection_names), total=len(list(collection_names))):
        pass


def prepare_job(model):
    client = MongoClient(CON_STRING)
    db_preprocessing = client.harddrive_preprocessing

    logging.info('Processing ' + model)
    if 1 not in safe_distinct(db_preprocessing, model, 'failure'):
        return
    serial_numbers = safe_distinct(db_preprocessing, model, 'serial_number')
    logging.info('Adding ' + str(len(serial_numbers)) + ' serial numbers for model ' + model)
    safe_write([{'model': model, 'serial_number': serial_number} for serial_number in serial_numbers], db_preprocessing,
               'tasks')
    logging.info('Processing ' + model + ' done.')


def preprocess_parallel(relevant_model_ids):
    local_client = MongoClient(CON_STRING)
    db_preprocessing = local_client.harddrive_preprocessing

    task_records = list(db_preprocessing['tasks'].find({"model": {"$in": relevant_model_ids}}))

    logging.info('Found ' + str(len(task_records)) + ' tasks.')

    p = Pool(6)
    for _ in tqdm.tqdm(p.imap_unordered(preprocess_job, task_records), total=len(task_records)):
        pass


def preprocess_job(task):
    local_client = MongoClient(CON_STRING)
    db_preprocessing = local_client.harddrive_preprocessing

    model = task.get('model')
    serial_number = task.get('serial_number')
    if model is None or serial_number is None:
        raise Exception('model or serial_number not set.')

    records = safe_find(db_preprocessing, model, 'serial_number', serial_number)
    df_records = pd.DataFrame(records)
    if 1 not in df_records['failure'].unique() or len(df_records.index) < 30:
        return
    df_records.rename(columns={'date': 'timestamp'}, inplace=True)
    df_records.drop('_id', axis=1, inplace=True)
    run_to_failures = []
    last_failure_date = None

    df_failures = df_records[df_records['failure'] == 1]
    df_failures = df_failures.sort_values('timestamp')
    df_failures = df_failures.reset_index()
    for idx, row in df_failures.iterrows():
        df_run_to_failure = None
        if last_failure_date is None:
            df_run_to_failure = df_records[df_records['timestamp'] <= row['timestamp']].copy()
        else:
            df_run_to_failure = df_records[
                (df_records['timestamp'] <= row['timestamp']) & (df_records['timestamp'] > last_failure_date)].copy()

        # Replace datetime timestamp with integer
        # df_run_to_failure.reset_index(inplace=True)
        # df_run_to_failure['timestamp_date'] = pd.to_datetime(df_run_to_failure['timestamp'], errors='raise', exact=True)
        # df_run_to_failure.drop('timestamp', inplace=True, axis=1)
        # df_run_to_failure.sort_values(by='timestamp_date', ascending=True, inplace=True)
        # min_timestamp = min(df_run_to_failure['timestamp_date'])
        # df_run_to_failure['timestamp'] = (df_run_to_failure['timestamp_date'].apply(lambda ts: (ts - min_timestamp).total_seconds() / 86400)).astype(int)
        # df_run_to_failure.drop('timestamp_date', inplace=True, axis=1)

        # drop normalized cols
        normalized_cols = [colname for colname in df_run_to_failure.columns if 'normalized' in colname]
        df_run_to_failure.drop(normalized_cols, axis=1, inplace=True)

        run_to_failures.append(df_run_to_failure)

        df_run_to_failure.drop(['failure', 'model', 'serial_number'], axis=1, inplace=True)
        records = [{k: v for k, v in item.items() if type(v) is str or math.isnan(v) is False} for item
                   in df_run_to_failure.to_dict('records')]
        if len(records) > 0:
            df_out = pd.DataFrame(records)
            df_out['timestamp'] = pd.to_datetime(df_out['timestamp'], errors="raise", exact=True)
            df_out.set_index('timestamp', inplace=True, drop=True)
            data_manager.write_rtf(dataset_id=dataset_id, model_id=model, equipment_id=serial_number, run_idx=idx,
                                   df=df_out)

        last_failure_date = row['timestamp']


def safe_distinct(db, collection, key):
    try:
        return db[collection].distinct(key)
    except Exception as e:
        logging.error(e)
        logging.info('Trying distinct again in 60 seconds...')
        time.sleep(60)
        return safe_distinct(db, collection, key)


def safe_find(db, model, key, value):
    try:
        if key is None or value is None:
            return list(db[model].find())
        else:
            return list(db[model].find({key: value}))
    except Exception as e:
        logging.error(e)
        logging.info('Trying find again in 60 seconds...')
        time.sleep(60)
        return safe_find(db, model, key, value)


def safe_write(records, db_out, collection_name):
    try:
        db_out[collection_name].insert_many(records)
        return None
    except Exception as e:
        logging.error(e)
        logging.info('Trying write again in 60 seconds...')
        time.sleep(60)
        return safe_write(records, db_out, collection_name)


if __name__ == '__main__':
    # Load raw data into database by collection
    # load_parallel()
    # Create index on preprocessing database
    # create_index()
    # Prepare tasks and write to mongo
    # prepare_parallel()

    CON_STRING = 'localhost:27017'

    # Cleanup
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_id = dir_path.split('/')[-1]

    data_manager = DataManager()

    if dataset_id in data_manager.dataset_ids:
        print('Deleting ' + dataset_id)
        data_manager.remove_dataset(dataset_id)

    # Execute jobs on tasks
    relevant_model_ids = ['WDC WD800AAJS',
                          'ST3160318AS',
                          'WDC WD20EFRX',
                          'ST9250315AS',
                          'ST3160316AS',
                          'WDC WD1600AAJB',
                          'ST320005XXXX',
                          'WDC WD800JB']
    preprocess_parallel(relevant_model_ids)
