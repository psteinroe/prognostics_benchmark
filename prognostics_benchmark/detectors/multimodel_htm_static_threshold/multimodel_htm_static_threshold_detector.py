import math
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from prognostics_benchmark import run_to_failure

from ..base import Detector as BaseDetector
from .univ_htm_detector import UnivHTMDetector

default_params = {
    "long_htm_probationary_period": 1227,
    "long_smoothing_kernel_size_ratio": 0.31284135997854223,
    "long_threshold": -200.0,
    "short_htm_probationary_period": 88,
    "short_smoothing_kernel_size_ratio": 0.1,
    "short_threshold": -48.29673293328383
}

default_config = {}


class MultiModelHTMDetectorStaticThreshold(BaseDetector):
    def __init__(self,
                 *args,
                 **kwargs):
        super(MultiModelHTMDetectorStaticThreshold, self).__init__(default_params=default_params, default_config=default_config, *args, **kwargs)

        self.univ_detectors_map = {}
        self._iteration = -1

        # We use this to set bounds and params
        self.first_ts = None
        self.use_short = None

    @staticmethod
    def get_default_params():
        return default_params

    @staticmethod
    def get_default_config():
        return default_config

    def handle_record(self, ts, data):
        self._iteration = self._iteration + 1

        if self.first_ts is None:
            self.first_ts = ts
            return {
                "is_alarm": False,
                "anomaly_score": 0,
                "threshold": 0
            }

        if self.use_short is None:
            # Transform to one dict with 5 keys and the orig. names
            if ts - self.first_ts < timedelta(hours=1):
                self.params = {
                    key.replace('long_', ''): value for key, value in self.params.items() if key.startswith('long_')
                }
                self.use_short = False
            else:
                self.params = {
                    key.replace('short_', ''): value for key, value in self.params.items() if key.startswith('short_')
                }
                self.use_short = True

        # Step 1: Gather individual scores
        scores = {}
        for colname, val in data.items():
            # Step 1.1: Get likelihood from individual model
            if colname not in self.univ_detectors_map:
                # create detector
                if self.verbose:
                    print('[{}] Creating univariate detector for {}'.format(self._iteration, colname))
                self.univ_detectors_map[colname] = UnivHTMDetector(
                    name=colname,
                    probationaryPeriod=self.params['htm_probationary_period'],
                    smoothingKernelSize=int(self.params['htm_probationary_period'] * self.params['smoothing_kernel_size_ratio']),
                    verbose=self.verbose)
                self.univ_detectors_map[colname].initialize()

            if np.isscalar(val) is True:  # check whether scalar to be able to handle null or missing values
                likelihood = self.univ_detectors_map[colname].modelRun(ts, val)
                scores[colname] = likelihood

        # Step 2: Combine individual scores
        log_scores = np.asarray([math.log(score) for score in list(scores.values())])
        anomaly_score = np.sum(log_scores)

        # Step 3: Apply Static Threshold
        if anomaly_score > self.params['threshold']:
            is_alarm = True
        else:
            is_alarm = False

        return {
            "is_alarm": is_alarm,
            "anomaly_score": anomaly_score,
            "threshold": self.params['threshold']
        }

    def _reset_params(self):
        self.univ_detectors_map = {}
        self._iteration = -1

    def failure_reached(self, run: run_to_failure):
        """
        Run param optimization on past failures, set new config and reset model
        :param run:
        :return:
        """
        self._reset_params()

    def get_additional_headers(self):
        return ['anomaly_score', 'threshold']

    def plot(self, df):
        """
        Plot the results given by the run

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with 'is_alarm', 'anomaly_score' and 'threshold' as attributes and the timestamp as index

        Returns
        ----------
        list
            list of the plots
        """
        if 'anomaly_score' not in df.columns or 'is_alarm' not in df.columns or 'threshold' not in df.columns:
            raise Exception('Invalid dataframe attributes.')

        deep_saffron = '#FF9933'
        air_force_blue = '#5D8AA8'

        x = df.index

        ts_fig, = plt.plot(x, df.anomaly_score, color=air_force_blue, label='Anomaly Score')
        fig = [ts_fig]

        th_fig, = plt.plot(x, df.threshold, color=deep_saffron, lw=2, ls='dashed', label='Threshold')
        fig.append(th_fig)

        if df.is_alarm[df['is_alarm']].size > 0:
            # x is alarm idx and y is anomaly score at that index
            df_alarms = df.loc[df['is_alarm']]
            plt.scatter(x=df_alarms.index, y=df_alarms.anomaly_score, color='red', label='Alarms Raised')

        plt.xlim((x[0], x[-1]))

        return fig
