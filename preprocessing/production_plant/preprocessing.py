import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append('../../')
from prognostics_benchmark.datasets.manager import DataManager


def get_runs():
    input_path = os.path.join("raw")

    runs = {}

    for in_file in os.scandir(input_path):
        name = in_file.name.split('.')[0]
        file_id = name.split('-')[0]

        df = pd.read_csv(in_file.path, header=0)
        df.rename(columns={'Timestamp': 'timestamp'}, inplace=True)
        df.set_index('timestamp', inplace=True)

        if file_id in runs:
            file_num = name.split('-')[1]
            if int(file_num) == 1:
                for i in range(len(df.index), len(df.index) + 100):
                    df.loc[i] = np.NaN
                df_merged = pd.concat([df, runs[file_id]], ignore_index=True, sort=True)
            elif int(file_num) == 2:
                for i in range(len(runs[file_id].index), len(runs[file_id].index) + 100):
                    runs[file_id].loc[i] = np.NaN
                df_merged = pd.concat([runs[file_id], df], ignore_index=True, sort=True)
            else:
                raise Exception('Unknown file number: ' + file_num)

            df_merged.reset_index(inplace=True)
            df_merged['timestamp'] = range(0, len(df_merged.index - 1))
            runs[file_id] = df_merged
        else:
            df.reset_index(inplace=True)
            runs[file_id] = df

    return runs


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_id = dir_path.split('/')[-1]

    data_manager = DataManager()

    if dataset_id in data_manager.dataset_ids:
        print('Deleting ' + dataset_id)
        data_manager.remove_dataset(dataset_id)

    dt = datetime.now()
    for file_id, df in get_runs().items():
        df['timedelta'] = df['timestamp'] * 5
        df['timestamp'] = df.apply(lambda row: dt + timedelta(seconds=row['timedelta']), axis=1)
        df.drop('timedelta', axis=1, inplace=True)
        df.set_index('timestamp', inplace=True, drop=True)
        run_idx = int(''.join(c for c in file_id if c.isdigit()))
        df.drop('index', axis=1, inplace=True, errors='ignore')
        data_manager.write_rtf(dataset_id=dataset_id, model_id=str(0), equipment_id=str(0), run_idx=run_idx, df=df)
