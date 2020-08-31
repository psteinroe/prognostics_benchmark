from collections import deque
import pandas as pd
import xgboost as xgb
import math
from datetime import timedelta

from ..base import Detector as BaseDetector
from ..utils import add_rul, align_columns

default_params = {
    'prediction_horizon_lead_time_ratio': 3,
}

default_config = {
    'buffer_size': 15
}


class XGBRegressorDetector(BaseDetector):

    def __init__(self, *args, **kwargs):
        super(XGBRegressorDetector, self).__init__(default_params=default_params, default_config=default_config, *args, **kwargs)

        self.prediction_horizon = self.evaluator.get_lead_time() * self.params['prediction_horizon_lead_time_ratio']
        self.model = None
        self.runs = deque(maxlen=self.config['buffer_size'])

    @staticmethod
    def get_default_params():
        return default_params

    def handle_record(self, ts, data):
        if self.model is None:
            return {
                "is_alarm": False
            }

        X_df = pd.DataFrame([data])
        X_df = align_columns(df=X_df, sorted_feature_names=self.model.get_booster().feature_names)

        prediction = self.model.predict(X_df)[0]
        if math.isnan(prediction) is True:
            return {
                "is_alarm": False
            }

        rul_pred = timedelta(seconds=int(prediction))

        if rul_pred < self.prediction_horizon:
            return {
                "is_alarm": True
            }
        else:
            return {
                "is_alarm": False
            }

    def failure_reached(self, run):
        self.runs.append(run)

        # Train on all runs
        df_all_runs = pd.DataFrame()
        for rtf in list(self.runs):
            df_run = rtf.get_df()
            df_run = add_rul(df_run)
            df_all_runs = pd.concat([df_all_runs, df_run], ignore_index=True, sort=True)

        features = [colname for colname in df_all_runs.columns if colname != 'rul']
        X_train = df_all_runs.loc[:, features]
        X_train = X_train.reindex(sorted(X_train.columns), axis=1)
        y_train = df_all_runs.loc[:, 'rul'].dt.total_seconds()

        self.model = xgb.XGBRegressor(
            max_depth=13,
            learning_rate=0.02,
            reg_alpha=1,
            reg_lambda=0)
        self.model.fit(X_train, y_train)
