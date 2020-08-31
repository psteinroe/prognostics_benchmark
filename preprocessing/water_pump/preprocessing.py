import os
import sys
import pandas as pd

sys.path.append('../../')
from prognostics_benchmark.datasets.manager import DataManager


class ApplyHelper(object):
    def __init__(self):
        self.ctr = 0

    def process_row(self, row):
        if row['machine_status'] == row['prev_status']:
            return self.ctr
        else:
            self.ctr = self.ctr + 1
            return self.ctr


def get_runs():
    input_path = os.path.join('raw')

    df_in = pd.read_csv(os.path.join(input_path, 'sensor.csv'), index_col=0)
    df_in['prev_status'] = df_in['machine_status'].shift(1)

    apply_helper = ApplyHelper()
    df_in['run'] = df_in.apply(lambda row: apply_helper.process_row(row), axis=1)

    run_to_failures = []
    grouped = df_in.groupby('run')
    number_of_runs = len(grouped)
    for (idx, (name, df)) in enumerate(grouped):
        if 'NORMAL' in df['machine_status'].unique() and idx != number_of_runs - 1:
            df_out = df.copy()
            df_out.drop(['machine_status', 'prev_status', 'run'], axis=1, inplace=True)
            df_out.reset_index(inplace=True, drop=True)
            df_out['timestamp'] = pd.to_datetime(df_out['timestamp'], errors='raise', exact=True)
            df_out.set_index('timestamp', inplace=True, drop=True)
            run_to_failures.append(df_out)
    return run_to_failures


def get_info():
    input_path = os.path.join('raw')

    df_in = pd.read_csv(os.path.join(input_path, 'sensor.csv'), index_col=0, parse_dates=['timestamp'])
    print(df_in.timestamp)
    print(df_in.timestamp.max() - df_in.timestamp.min())


if __name__ == '__main__':
    get_info()
    # dir_path = os.path.dirname(os.path.realpath(__file__))
    # dataset_id = dir_path.split('/')[-1]
    #
    # data_manager = DataManager()
    #
    # if dataset_id in data_manager.dataset_ids:
    #     print('Deleting ' + dataset_id)
    #     data_manager.remove_dataset(dataset_id)
    #
    # for idx, df_run in enumerate(get_runs()):
    #     cols_with_all_nan = df_run.columns[df_run.isnull().all()].tolist()
    #     if len(cols_with_all_nan) > 0:
    #         print('Dropping ' + ', '.join(cols_with_all_nan))
    #         df_run.drop(cols_with_all_nan, inplace=True, axis=1)
    #
    #     data_manager.write_rtf(dataset_id=dataset_id, model_id=str(0), equipment_id=str(0), run_idx=idx, df=df_run)
