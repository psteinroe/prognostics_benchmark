from datetime import timedelta
import matplotlib.pyplot as plt

from prognostics_benchmark import run_to_failure

from ..base import Detector as BaseDetector
from .multiv_htm_detector import MultivHTMDetector
from .online_drift_spot import OnlineDriftSpot
from ..utils import dot_to_json

default_params = {
    'htm.anomaly.likelihood.probationaryPct': 0.417022004702574,
    'htm.anomaly.likelihood.reestimationPeriod': 100,
    'htm.enc.time.timeOfDay': (21, 6.456740123240503),
    'htm.enc.time.value.activeBits': 23,
    'htm.enc.time.value.size': 400,
    'htm.enc.time.value.seed': 0,
    'htm.sp.boostStrength': 0.0,
    'htm.sp.wrapAround': True,
    'htm.sp.columnDimensions': 7231,
    'htm.sp.dutyCyclePeriod': 101,
    'htm.sp.minPctOverlapDutyCycle': 0.30233257263183977,
    'htm.sp.localAreaDensity': 0,
    'htm.sp.numActiveColumnsPerInhArea': 77,
    'htm.sp.potentialPct': 0.0923385947687978,
    'htm.sp.globalInhibition': True,
    'htm.sp.stimulusThreshold': 93,
    'htm.sp.synPermActiveInc': 0.34556072704304774,
    'htm.sp.synPermConnected': 0.39676747423066994,
    'htm.sp.synPermInactiveDec': 0.538816734003357,
    'htm.sp.seed': 0,
    'htm.tm.activationThreshold': 212,
    'htm.tm.cellsPerColumn': 344,
    'htm.tm.connectedPermanence': 0.20445224973151743,
    'htm.tm.initialPermanence': 0.8781174363909454,
    'htm.tm.maxNewSynapseCount': 32,
    'htm.tm.maxSegmentsPerCell': 672,
    'htm.tm.maxSynapsesPerSegment': 420,
    'htm.tm.minThreshold': 281,
    'htm.tm.permanenceDecrement': 0.14038693859523377,
    'htm.tm.permanenceIncrement': 0.1981014890848788,
    'htm.tm.predictedSegmentDecrement': 0.8007445686755367,
    'htm.tm.seed': 0,
    "short_htm_probationary_period": 91,
    "short_spot_depth_ratio": 0.14252210568488896,
    "short_spot_probationary_period_ratio": 0.13514930490959415,
    "short_spot_q": 0.084998226740328,
    "long_htm_probationary_period": 4844,
    "long_spot_depth_ratio": 0.2567120890796214,
    "long_spot_probationary_period_ratio": 0.7230903541023826,
    "long_spot_q": 0.43820693723278953,
}

default_config = {}


class UniModelHTMDetector(BaseDetector):
    def __init__(self,
                 *args,
                 **kwargs):
        super(UniModelHTMDetector, self).__init__(default_params=default_params, default_config=default_config, *args, **kwargs)

        self.htm_model = None
        self._iteration = 0

        # Drift SPOT
        self.drift_spot = None

        self.use_short = None
        self.first_ts = None

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
            # Transform params and bounds
            if ts - self.first_ts < timedelta(hours=1):
                self.params = {
                    **{key.replace('long_', ''): value for key, value in self.params.items() if key.startswith('long_')},
                    **{key: value for key, value in self.params.items() if not key.startswith('long_') and not key.startswith('short_')}
                }
                self.use_short = False
            else:
                self.params = {
                    **{key.replace('short_', ''): value for key, value in self.params.items() if key.startswith('short_')},
                    **{key: value for key, value in self.params.items() if not key.startswith('short_') and not key.startswith('long_')},
                }
                self.use_short = True

        # Step 1: Send through HTM Model
        if self.htm_model is None:
            if self.verbose:
                print('[{}] Creating HTM Model'.format(self._iteration))
            params_dict = dot_to_json(self.params)
            self.htm_model = MultivHTMDetector(
                name='UniModel',
                probationaryPeriod=self.params.get('htm_probationary_period'),
                params=params_dict.get('htm'),
                verbose=self.verbose
            )

        anomaly_score, raw = self.htm_model.modelRun(ts, data)
        if self.verbose:
            print('[{}] HTM Raw: {}'.format(self._iteration, raw))
            print('[{}] HTM Likelihood: {}'.format(self._iteration, anomaly_score))

        # Step 2: Feed to Online Drift SPOT
        if self._iteration < self.params['htm_probationary_period'] + 2:
            return {
                "is_alarm": False,
                "anomaly_score": anomaly_score,
                "threshold": anomaly_score  # return anomaly score as threshold during SPOT initialization
            }

        if self.drift_spot is None:
            if self.verbose:
                print('[{}] Creating SPOT'.format(self._iteration))
            self.drift_spot = OnlineDriftSpot(
                probationary_period=int(self.params['htm_probationary_period'] * self.params['spot_probationary_period_ratio']),
                q=self.params['spot_q'],
                depth_ratio=self.params['spot_depth_ratio'],
                verbose=self.verbose)

        if self.verbose:
            print('[{}] Feeding to Drift SPOT: {}'.format(self._iteration, anomaly_score))
        is_alarm, threshold = self.drift_spot.add(anomaly_score)

        return {
            "is_alarm": is_alarm,
            "anomaly_score": anomaly_score,
            "threshold": threshold
        }

    def _reset_params(self):
        self.htm_model = None
        self._iteration = 0
        self.drift_spot = None

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
