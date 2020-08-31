from ..base import Detector as BaseDetector

default_params = {}

default_config = {}


class RunToFailureDetector(BaseDetector):
    """
    Detector resembling the Run to Failure Maintenance Strategy
    """
    def __init__(self,
                 *args,
                 **kwargs):
        super(RunToFailureDetector, self).__init__(default_params=default_params, default_config=default_config, *args, **kwargs)

    def handle_record(self, ts, data):
        return {
            "is_alarm": False
        }

    @staticmethod
    def get_default_params():
        return {}
