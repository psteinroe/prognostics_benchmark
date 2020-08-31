from ..base import Detector as BaseDetector

default_params = {}

default_config = {}


class AlwaysTrueDetector(BaseDetector):
    """
    Detector that always returns true.
    """
    def __init__(self,
                 *args,
                 **kwargs):
        super(AlwaysTrueDetector, self).__init__(default_params=default_params, default_config=default_config, *args, **kwargs)

    def handle_record(self, ts, data):
        return {
            "is_alarm": True
        }

    @staticmethod
    def get_default_params():
        return {}
