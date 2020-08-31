from math import ceil
import matplotlib.pyplot as plt
from datetime import timedelta

from ..base import Evaluator

default_config = {
    'harddrive': {
        'lead_time': timedelta(days=4),
        'relevant_maintenance_factor': 5,
        'cost_rate': 20,
    },
    'turbofan_engine': {
        'lead_time': timedelta(days=5),
        'relevant_maintenance_factor': 3,
        'cost_rate': 50.33,
    },
    'water_pump': {
        'lead_time': timedelta(hours=36),
        'relevant_maintenance_factor': 3,
        'cost_rate': 7.01,
    },
    'production_plant': {
        'lead_time': timedelta(hours=2),
        'relevant_maintenance_factor': 5,
        'cost_rate': 18.13,
    }
}


class MaintenanceCostEvaluator(Evaluator):

    def __init__(self, *args, **kwargs):
        super(MaintenanceCostEvaluator, self).__init__(default_config=default_config, *args, **kwargs)

        self.lead_time = self.config['lead_time']

        self.relevant_maintenance_window_size = self.lead_time * self.config['relevant_maintenance_factor']
        self.alert_deduplication_window = self.config['lead_time'] * 0.75  # after an alarm is raised, we ignore alarms for this time

        self.maintenance_cost = 1000
        self.cost_of_failure = self.maintenance_cost * self.config['cost_rate']

    @staticmethod
    def get_default_config():
        return default_config

    def evaluate_rtf(self, df, plot=False):
        # Validity Check: Return 100 if lead time is larger than the RTF
        if df.index.max() - df.index.min() <= self.lead_time:
            return {
                'evaluation': 100,
                'failure_prevented': True,
                'n_maintenance_activities': 0,
            }

        # 1. Get timestamps and dfs for cutoffs of lead time and relevant maintenance window
        x_lead_time_begin = df.index.max() - self.lead_time
        x_relevant_maintenance_begins = x_lead_time_begin - self.relevant_maintenance_window_size

        # 2. Get max and min costs for this rtf
        min_cost = self.maintenance_cost  # one repair activity

        td_to_relevant_maintenance = max(timedelta(0), x_relevant_maintenance_begins - df.index.min())
        max_non_relevant_activities = ceil(td_to_relevant_maintenance / self.alert_deduplication_window) + ceil(self.lead_time / self.alert_deduplication_window)
        max_cost = max(
            (max_non_relevant_activities + 1) * self.maintenance_cost,
            # all non-relevant activities +1 one repair visit
            max_non_relevant_activities * self.maintenance_cost + self.cost_of_failure
            # all non-relevant activities + machine fails
        )

        # 3. Get all alarms and prune them by applying de-duplication
        df_alarms = df[df['is_alarm']]
        x_maintenance_activities = []
        prev_alarm_at = None
        for idx, row in df_alarms.iterrows():
            if prev_alarm_at is None or row.name - prev_alarm_at > self.alert_deduplication_window:
                x_maintenance_activities.append(row.name)
                prev_alarm_at = row.name

                if x_lead_time_begin >= row.name >= x_relevant_maintenance_begins:
                    # Machine got fixed
                    break

        cost = len(x_maintenance_activities) * self.maintenance_cost

        # 4. If machine failed, add costs of failure
        failure_prevented = False
        if len([timestamp for timestamp in x_maintenance_activities if
                x_relevant_maintenance_begins <= timestamp <= x_lead_time_begin]) == 0:
            cost = cost + self.cost_of_failure
        else:
            failure_prevented = True

        # 5. Normalize cost
        normalized = round(100 - (cost - min_cost) / (max_cost - min_cost) * 100, 4)

        # Plotting functionality
        if plot is True:
            ax = plt.gca()
            ax.axvline(x=x_lead_time_begin, color='red', label='Begin Lead Time', ls='--')
            ax.axvline(x=x_relevant_maintenance_begins, color='blue', label='Begin Relevant Maintenance', ls='--')
            maintenance_visits = [x for x in x_maintenance_activities if x <= x_relevant_maintenance_begins]
            maintenance_repairs = [x for x in x_maintenance_activities if x_relevant_maintenance_begins <= x <= x_lead_time_begin]
            ymin, ymax = plt.gca().get_ylim()
            if len(maintenance_visits) > 0:
                ax.vlines(x=maintenance_visits, ymin=ymin, ymax=ymax, color='orange', label='Maintenance Visits')
            if len(maintenance_repairs) > 0:
                ax.vlines(x=maintenance_repairs, ymin=ymin, ymax=ymax, color='green', label='Maintenance Repair')
            return {
                'evaluation': normalized,
                'failure_prevented': failure_prevented,
                'n_maintenance_activities': len(x_maintenance_activities),
                'plot': plt.gcf()
            }

        return {
            'evaluation': normalized,
            'failure_prevented': failure_prevented,
            'n_maintenance_activities': len(x_maintenance_activities),
        }

    def combine_rtf_evaluations(self, rtf_scores):
        # just return normalized score
        max_score = 100 * len(rtf_scores)
        min_score = 0
        return round((sum(rtf_scores) - min_score) / (max_score - min_score) * 100, 4)

    @staticmethod
    def combine_dataset_evaluations(dataset_scores):
        # just return normalized score
        max_score = 100 * len(dataset_scores)
        min_score = 0
        return round((sum(dataset_scores) - min_score) / (max_score - min_score) * 100, 4)
