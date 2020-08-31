import os
import sys
from datetime import datetime, timedelta
import pandas as pd

sys.path.append('../../')
from prognostics_benchmark.datasets.manager import DataManager

DATASET_NAMES = ['FD001', 'FD002', 'FD003', 'FD004']

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_id = dir_path.split('/')[-1]

    data_manager = DataManager()

    if dataset_id in data_manager.dataset_ids:
        print('Deleting ' + dataset_id)
        data_manager.remove_dataset(dataset_id)

    input_path = os.path.join('raw')

    cols_index = ['UnitNumber', 'timestamp']
    cols_settings = ['OpSetting1', 'OpSetting2', 'OpSetting3']
    cols_sensors = ['Sensor' + str(i) for i in range(1, 22)]
    cols = cols_index + cols_settings + cols_sensors

    for name in DATASET_NAMES:
        df_train = pd.read_csv(os.path.join(input_path, 'train_' + name + '.txt'), header=None, sep=" ").iloc[:,
                   :26].copy()
        df_train.columns = cols

        for unit_number, df in df_train.groupby('UnitNumber'):
            df_out = df.copy()
            date_today = datetime.now()
            days = pd.date_range(date_today-timedelta(days=len(df_out.index)-1), date_today, freq='D', normalize=True)
            df_out.set_index(days, inplace=True)
            df_out.drop(['timestamp', 'UnitNumber'], inplace=True, axis=1)
            data_manager.write_rtf(dataset_id, str(name), str(unit_number), 0, df_out)
