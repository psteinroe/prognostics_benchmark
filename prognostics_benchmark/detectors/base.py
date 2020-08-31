from typing import TYPE_CHECKING, Tuple
if TYPE_CHECKING:
    from prognostics_benchmark.run_to_failure import RunToFailure

from datetime import datetime
import pandas as pd

from prognostics_benchmark.evaluators.base import Evaluator


class Detector(object):

    def __init__(self,
                 evaluator: Evaluator,
                 default_params: dict,
                 default_config: dict,
                 params: dict = None,
                 config: dict = None,
                 verbose: bool = False):
        """
          Base class for all algorithms. When inheriting from this class please
          take note of which methods MUST be overridden, as documented below.
        """
        self.evaluator = evaluator
        self.params = params
        self.config = config
        self.verbose = verbose

        if self.params is None:
            self.params = default_params

        if set(self.params.keys()) != set(default_params.keys()):
            raise Exception('Missing some parameters.')

        if self.config is None:
            self.config = default_config

        if set(self.config.keys()) != set(default_config.keys()):
            raise Exception('Missing some config parameters.')

    @staticmethod
    def get_default_params() -> dict:
        """
        Must be implemented within the subclass. Simply returns the default parameters.
        This method MUST be overridden by subclasses.
        :return: default parameters
        """
        raise NotImplementedError

    def handle_record(self, ts: datetime, data: pd.Series) -> dict:
        """
        Returns a dict which must contain at least a boolean value for 'is_alarm' that indicates whether or not
        an alarm should be raised for the given data point.
        Input is a timestamp ts and a Pandas Series consisting of 1 to n items.
        This method MUST be overridden by subclasses.
        :param ts: timestamp
        :param data: data
        :return: dictionary containing at least a boolean value for 'is_alarm'
        """
        raise NotImplementedError

    def failure_reached(self, rtf: 'RunToFailure') -> None:
        """
            Do anything you want when a failure is reached, e.g. retrain or optimize the model.
            Input is the RunToFailure processed previously.
        """
        pass

    @staticmethod
    def plot(df: pd.DataFrame, *args, **kwargs):
        """
            Plot the results_supervised given by the run. Given is a pandas dataframe with at least 'is_alarm' as attribute.
        """
        return df[df['is_alarm']].reset_index().plot.scatter(x='timestamp',
                                                             y='is_alarm',
                                                             include_bool=True,
                                                             ylim=(0.5, 1.5),
                                                             xlim=(df.iloc[0].name, df.iloc[-1].name),
                                                             color='red',
                                                             *args,
                                                             **kwargs)
