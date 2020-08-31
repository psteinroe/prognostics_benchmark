from tqdm import tqdm
import pandas as pd
from datetime import datetime

from prognostics_benchmark.detectors.base import Detector as BaseDetector
from prognostics_benchmark.datasets import DataManager
from prognostics_benchmark.utils import parse_rtf_id


class RunToFailure:
    def __init__(self, dataset_id: str, rtf_id: str):
        self.dataset_id: str = dataset_id
        self.id: str = rtf_id

        self.model_id: str
        self.equi_id: str
        self.run_idx: int
        self.model_id, self.equi_id, self.run_idx = parse_rtf_id(rtf_id)

        self.data_manager: DataManager = DataManager()

    def get_df(self) -> pd.DataFrame:
        return self.data_manager.load_rtf(dataset_id=self.dataset_id, rtf_id=self.id)

    def score(self, detector, verbose: bool = False) -> pd.DataFrame:
        df = self.get_df().copy()

        if isinstance(detector, BaseDetector) is False:
            raise Exception('Given instance is not of class Detector.')

        results = []

        for idx, row in tqdm(df.iterrows(), total=len(df.index), disable=not verbose):
            start_time = datetime.now()
            res = detector.handle_record(row.name, row)
            processing_time = datetime.now() - start_time

            if "is_alarm" not in res or type(res.get("is_alarm")) != bool:
                raise Exception('Dictionary must at least contain a value "is_alarm" of type bool.')
            if "timestamp" in res:
                raise Exception('Key "timestamp" is reserved.')
            if "processing_time" in res:
                raise Exception('Key "processing_time" is reserved.')
            if "rtf_id" in res:
                raise Exception('Key "rtf_id" is reserved.')

            res['timestamp'] = row.name
            res['processing_time'] = processing_time

            results.append(res)

        detector.failure_reached(self)

        df_results = pd.DataFrame(results)
        # Add ids for later reference
        df_results['rtf_id'] = self.id
        df_results.set_index('timestamp', inplace=True, drop=True)

        return df_results
