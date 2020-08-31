import os
import pandas as pd
from pathlib import Path
import numpy as np
import shutil

from prognostics_benchmark.utils import parse_rtf_id


def sorted_rtf_ids(rtf_ids_unsorted: [str]) -> [str]:
    return sorted(rtf_ids_unsorted, key=lambda rtf_id: parse_rtf_id(rtf_id))


def read_rtf_ids_from_dir(path: str) -> [str]:
    return [filename.replace('.csv', '') for filename in os.listdir(path) if filename.endswith(".csv")]


class DataManager:
    def __init__(self):
        full_path: str = os.path.realpath(__file__)
        self.my_dir_path: str = os.path.dirname(full_path)

    @property
    def dataset_ids(self) -> [str]:
        return [
            os.path.join(self.my_dir_path, o).split('/')[-1]
            for o in os.listdir(self.my_dir_path)
            if os.path.join(self.my_dir_path, o).split('/')[-1] != '__pycache__'
               and os.path.isdir(os.path.join(self.my_dir_path, o))
        ]

    def remove_dataset(self, dataset_id: str) -> None:
        shutil.rmtree(os.path.join(self.my_dir_path, dataset_id))

    def write_rtf(self, dataset_id: str, model_id: str, equipment_id: str, run_idx: int, df: pd.DataFrame) -> None:
        if any(np.isnat(df.index, dtype="datetime64[ns]")):
            raise Exception('Index must be of type datetime.')
        if len(df.index) < 100:
            raise Exception('The minimum length for a run to failure is 100.')
        if "_" in model_id or "_" in equipment_id or type(run_idx) != int:
            raise Exception("Substring '_' is not allowed and run_idx must be integer.")

        filename = model_id + '_' + equipment_id + '_' + str(run_idx) + '.csv'
        model_dir_path = os.path.join(self.my_dir_path, dataset_id, model_id)

        Path(model_dir_path).mkdir(parents=True, exist_ok=True)
        df.to_csv(os.path.join(model_dir_path, filename))

    def get_rtf_ids(self, dataset_id: str) -> [str]:
        data_path = os.path.join(self.my_dir_path, dataset_id)
        ids = []
        for model_path in [os.path.join(data_path, o) for o in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, o))]:
            rtf_ids = read_rtf_ids_from_dir(model_path)
            ids = ids + sorted_rtf_ids(rtf_ids)

        return ids

    def get_rtf_ids_for_model(self, dataset_id: str, model_id: str) -> [str]:
        data_path = os.path.join(self.my_dir_path, dataset_id)
        model_rtf_ids = read_rtf_ids_from_dir(os.path.join(data_path, model_id))
        return sorted_rtf_ids(model_rtf_ids)

    def get_rtf_ids_for_equipment(self, dataset_id: str, model_id: str, equipment_id: str) -> [str]:
        model_rtf_ids = self.get_rtf_ids_for_model(dataset_id=dataset_id, model_id=model_id)
        return [rtf_id for rtf_id in model_rtf_ids if parse_rtf_id(rtf_id)[1] == equipment_id]

    def get_model_ids(self, dataset_id: str) -> [str]:
        data_path = os.path.join(self.my_dir_path, dataset_id)
        return [model_path.split('/')[-1] for model_path in reversed(os.listdir(data_path)) if
                os.path.isdir(os.path.join(data_path, model_path))]

    def get_equipment_ids(self, dataset_id: str, model_id: str) -> [str]:
        rtf_ids = self.get_rtf_ids_for_model(dataset_id, model_id)
        return [parse_rtf_id(rtf_id)[1] for rtf_id in rtf_ids if parse_rtf_id(rtf_id)[0] == model_id]

    def load_rtf(self, dataset_id: str, rtf_id: str) -> pd.DataFrame:
        data_path = os.path.join(self.my_dir_path, dataset_id)
        model_id, _, _ = parse_rtf_id(rtf_id)
        df = pd.read_csv(os.path.join(data_path, str(model_id), str(rtf_id) + '.csv'), index_col=0, parse_dates=True, date_parser=np.datetime64)
        df.index.name = 'timestamp'
        df.sort_index(inplace=True)
        return df
