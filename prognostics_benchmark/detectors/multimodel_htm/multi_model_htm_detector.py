import math
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import deque
from bayes_opt import BayesianOptimization

from prognostics_benchmark import run_to_failure

from ..base import Detector as BaseDetector
from .univ_htm_detector import UnivHTMDetector
from .online_drift_spot import OnlineDriftSpot

default_params = {
    "long_htm_probationary_period": 587,
    "long_smoothing_kernel_size_ratio": 0.7414272592231331,
    "long_spot_depth_ratio": 0.281383440499545,
    "long_spot_probationary_period_ratio": 0.48675581242120003,
    "long_spot_q": 0.15703930922381432,
    "short_htm_probationary_period": 40,
    "short_smoothing_kernel_size_ratio": 0.7156307669706911,
    "short_spot_depth_ratio": 0.33113879781620026,
    "short_spot_probationary_period_ratio": 0.6696316716068865,
    "short_spot_q": 0.49733141622786337
}

default_config = {
    'optimize_on_failure': False,
    'optimization_init_points': 5,
    'optimization_n_iter': 15,
    'd_spot_update_threshold_on_alarm': True,
    'reset_after_rtf': True
}


class MultiModelHTMDetector(BaseDetector):
    def __init__(self,
                 *args,
                 **kwargs):
        super(MultiModelHTMDetector, self).__init__(default_params=default_params, default_config=default_config, *args, **kwargs)

        if self.config['optimize_on_failure'] is True and self.config['reset_after_rtf'] is False:
            raise ValueError('Can not optimize on failure without resetting after every RtF.')

        self.optimizer = None

        self.p_bounds = {
            'short_htm_probationary_period': (65, 90),
            'short_spot_probationary_period_ratio': (0.2, 1),
            'short_smoothing_kernel_size_ratio': (0.2, 1),
            'short_spot_q': (0.0001, 0.5),
            'short_spot_depth_ratio': (0.1, 0.5),
            'long_htm_probationary_period': (106, 4474),
            'long_spot_probationary_period_ratio': (0.2, 1),
            'long_smoothing_kernel_size_ratio': (0.2, 1),
            'long_spot_q': (0.01, 0.43),
            'long_spot_depth_ratio': (0.1, 0.27),
        }

        self.drift_spot = None
        self.univ_detectors_map = {}
        self._iteration = -1

        self.runs = deque(maxlen=20)

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
                "threshold": 0,
            }

        if self.use_short is None:
            # Transform to one dict with 5 keys and the orig. names
            if ts - self.first_ts < timedelta(hours=1):
                self.params = {
                    key.replace('long_', ''): value for key, value in self.params.items() if key.startswith('long_')
                }
                self.p_bounds = {
                    key.replace('long_', ''): value for key, value in self.p_bounds.items() if key.startswith('long_')
                }
                self.use_short = False
            else:
                self.params = {
                    key.replace('short_', ''): value for key, value in self.params.items() if key.startswith('short_')
                }
                self.p_bounds = {
                    key.replace('short_', ''): value for key, value in self.p_bounds.items() if key.startswith('short_')
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

        # Step 3: Feed to Online Drift SPOT
        if self._iteration < self.params['htm_probationary_period'] + int(self.params['htm_probationary_period'] * self.params['smoothing_kernel_size_ratio']):
            # univ htm probationary period + kernel size
            return {
                "is_alarm": False,
                "anomaly_score": anomaly_score,
                "threshold": anomaly_score,   # return anomaly score as threshold during SPOT initialization
                **scores
            }

        if self.drift_spot is None:
            if self.verbose:
                print('[{}] Creating SPOT'.format(self._iteration))
            self.drift_spot = OnlineDriftSpot(probationary_period=int(self.params['htm_probationary_period'] * self.params['spot_probationary_period_ratio']),
                                              q=self.params['spot_q'],
                                              depth_ratio=self.params['spot_depth_ratio'],
                                              update_threshold_on_alarm=self.config['d_spot_update_threshold_on_alarm'],
                                              verbose=self.verbose)
        is_alarm, threshold = self.drift_spot.add(anomaly_score)

        return {
            "is_alarm": is_alarm,
            "anomaly_score": anomaly_score,
            "threshold": threshold,
            **scores
        }

    def _reset_params(self, params=None):
        self.drift_spot = None
        self.univ_detectors_map = {}
        self._iteration = -1

        default_params_keys = set({
             key.replace('short_', ''): value for key, value in default_params.items() if key.startswith('short_')
         }.keys())

        if params is not None:
            if set(params.keys()) != default_params_keys:
                raise Exception('Missing some parameters.')

            self.params = params

    def failure_reached(self, run: run_to_failure):
        """
        Run param optimization on past failures, set new config and reset model
        :param run:
        :return:
        """
        if self.config['reset_after_rtf'] is False:
            return
        if self.config['optimize_on_failure'] is False:
            self._reset_params()
            return

        self.runs.append(run)

        detector = self

        def optimize(htm_probationary_period,
                     spot_probationary_period_ratio,
                     smoothing_kernel_size_ratio,
                     spot_q,
                     spot_depth_ratio):

            # Does not matter if long or short since we are working on the same dataset
            local_params = {
                'short_htm_probationary_period': int(htm_probationary_period),
                'short_spot_probationary_period_ratio': spot_probationary_period_ratio,
                'short_smoothing_kernel_size_ratio': smoothing_kernel_size_ratio,
                'short_spot_q': spot_q,
                'short_spot_depth_ratio': spot_depth_ratio,
                'long_htm_probationary_period': int(htm_probationary_period),
                'long_spot_probationary_period_ratio': spot_probationary_period_ratio,
                'long_smoothing_kernel_size_ratio': smoothing_kernel_size_ratio,
                'long_spot_q': spot_q,
                'long_spot_depth_ratio': spot_depth_ratio,
            }

            local_config = {
                'optimize_on_failure': False,
                'optimization_init_points': 5,
                'optimization_n_iter': 15,
            }

            local_detector = MultiModelHTMDetector(evaluator=detector.evaluator, params=local_params, config=local_config)

            scores = []
            for rtf in detector.runs:
                df_score = rtf.score(local_detector)
                rtf_evaluation = local_detector.evaluator.evaluate_rtf(df_score)
                scores.append(rtf_evaluation["evaluation"])

            return local_detector.evaluator.combine_rtf_evaluations(scores)

        if self.optimizer is None:
            self.optimizer = BayesianOptimization(
                f=optimize,
                pbounds=self.p_bounds,
                random_state=1,
                verbose=0
            )

        # Make sure we are always using the best
        self.optimizer.probe(
            params=self.params,
            lazy=True
        )

        self.optimizer.maximize(
            init_points=self.config['optimization_init_points'],
            n_iter=self.config['optimization_n_iter'],
        )

        self._reset_params({
            'htm_probationary_period': int(self.optimizer.max.get('params').get('htm_probationary_period')),
            'spot_probationary_period_ratio': self.optimizer.max.get('params').get('spot_probationary_period_ratio'),
            'smoothing_kernel_size_ratio': self.optimizer.max.get('params').get('smoothing_kernel_size_ratio'),
            'spot_q': self.optimizer.max.get('params').get('spot_q'),
            'spot_depth_ratio': self.optimizer.max.get('params').get('spot_depth_ratio'),
        })

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
