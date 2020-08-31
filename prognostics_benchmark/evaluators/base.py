from datetime import timedelta

import pandas as pd


class Evaluator:
    def __init__(self, dataset_id: str, default_config: dict, config: dict = None, verbose: bool = False):
        self.dataset_id = dataset_id
        self.config = config
        self.verbose = verbose

        if self.config is None:
            if self.dataset_id not in default_config.keys():
                raise Exception('Default config for given dataset id not found.')
            self.config = default_config.get(self.dataset_id)
        else:
            if self.dataset_id not in config.keys():
                raise Exception('Config for given dataset id not found.')
            self.config = config.get(self.dataset_id)

        if set(self.config.keys()) != set(default_config.get(self.dataset_id).keys()):
            raise Exception('Missing some parameters.')

        if self.config.get('lead_time') is None:
            raise Exception('Must provide lead time as parameter.')
        if type(self.config.get('lead_time')) != timedelta:
            raise Exception('lead_time must be of type timedelta')

    @staticmethod
    def get_default_config() -> dict:
        raise NotImplementedError

    def get_lead_time(self) -> timedelta:
        return self.config.get('lead_time')

    def evaluate_rtf(self, df: pd.DataFrame) -> dict:
        raise NotImplementedError

    def combine_rtf_evaluations(self, rtf_scores: [any]) -> any:
        raise NotImplementedError

    @staticmethod
    def combine_dataset_evaluations(dataset_scores: [any]) -> any:
        raise NotImplementedError
