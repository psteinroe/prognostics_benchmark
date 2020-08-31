from datetime import timedelta
from sklearn.metrics import f1_score
import pandas as pd
import statistics
import matplotlib.pyplot as plt

from ..base import Evaluator

default_config = {
    'harddrive': {
        'lead_time': timedelta(days=2),
        'prediction_horizon_factor': 6,
    },
    'turbofan_engine': {
        'lead_time': timedelta(days=8),
        'prediction_horizon_factor': 2,
    },
    'water_pump': {
        'lead_time': timedelta(hours=36),
        'prediction_horizon_factor': 2,
    },
    'production_plant': {
        'lead_time': timedelta(hours=8),
        'prediction_horizon_factor': 2,
    }
}


class SimpleWindowBasedEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        """
        :param relevant_maintenance_fraction: The fraction of the dataset length before the lead time in which maintenance activities prevent a failure.
        :param fn_rate: The rate in which not preventing the failure is more costly than doing a maintenance activity.
        :param args:
        :param kwargs:
        """
        super(SimpleWindowBasedEvaluator, self).__init__(default_config=default_config, *args, **kwargs)

        self.lead_time = self.config['lead_time']
        self.ph = self.config['lead_time'] * self.config['prediction_horizon_factor']

    @staticmethod
    def get_default_config():
        return default_config

    def evaluate_rtf(self, df, plot=False):
        ts_lead_time_begin = df.index.max() - self.lead_time
        ts_ph_begin = ts_lead_time_begin - self.ph
        df['true'] = df.apply(lambda row: ts_ph_begin < row.name <= ts_lead_time_begin, axis=1)

        evaluation = 1.0  # Validity Check: Return 1 if lead time is larger than the RTF
        if len(pd.unique(df['true'])) == 2:
            evaluation = f1_score(df['true'], df['is_alarm'])

        if plot:
            ax = plt.gca()
            ax.axvline(x=ts_lead_time_begin, color='red', label='Begin Lead Time', ls='--')
            ax.axvline(x=ts_ph_begin, color='blue', label='Begin Relevant Maintenance', ls='--')

        return {
            'evaluation': evaluation
        }

    def combine_rtf_evaluations(self, rtf_scores):
        return statistics.mean(rtf_scores)

    @staticmethod
    def combine_dataset_evaluations(dataset_scores):
        return statistics.mean(dataset_scores)
