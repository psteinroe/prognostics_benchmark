from tqdm import tqdm
from typing import Type
from datetime import timedelta
from prognostics_benchmark.detectors.base import Detector as BaseDetector
from prognostics_benchmark.evaluators.base import Evaluator as BaseEvaluator
from prognostics_benchmark.datasets import DataManager
from .run_to_failure import RunToFailure


class Dataset:
    def __init__(self, dataset_id: str):
        self.id: str = dataset_id

        self.dataset_manager: DataManager = DataManager()

    @property
    def rtf_ids(self) -> [str]:
        return self.dataset_manager.get_rtf_ids(dataset_id=self.id)

    def get_rtf_ids_for_model(self, model_id: str) -> [str]:
        if model_id not in self.model_ids:
            raise Exception('Model ID ' + model_id + ' invalid.')
        return self.dataset_manager.get_rtf_ids_for_model(dataset_id=self.id, model_id=model_id)

    def get_rtf_ids_for_equipment(self, model_id: str, equipment_id: str) -> [str]:
        if model_id not in self.model_ids:
            raise Exception('Model ID ' + model_id + ' invalid.')
        if equipment_id not in self.get_equipment_ids_for_model(model_id=model_id):
            raise Exception('Equipment ID ' + equipment_id + ' invalid.')
        return self.dataset_manager.get_rtf_ids_for_equipment(dataset_id=self.id, model_id=model_id,
                                                              equipment_id=equipment_id)

    @property
    def model_ids(self) -> [str]:
        return self.dataset_manager.get_model_ids(dataset_id=self.id)

    def get_equipment_ids_for_model(self, model_id: str) -> [str]:
        if model_id not in self.model_ids:
            raise Exception('Model ID ' + model_id + ' invalid.')
        return self.dataset_manager.get_equipment_ids(dataset_id=self.id, model_id=model_id)

    def get_rtf(self, rtf_id: str) -> RunToFailure:
        if rtf_id not in self.rtf_ids:
            raise Exception('Run ID ' + rtf_id + ' invalid.')

        return RunToFailure(self.id, rtf_id)

    def evaluate_on_models(self,
                           Detector: Type[BaseDetector],
                           evaluator: BaseEvaluator,
                           model_ids: [str] = None,
                           detector_params: dict = None,
                           detector_config: dict = None,
                           verbosity: int = 0):
        """

        :param Detector:
        :param evaluator:
        :param model_ids:
        :param detector_params:
        :param detector_config:
        :param verbosity:
        :return:
        """
        if isinstance(Detector, BaseDetector) is True:
            raise Exception('Detector is initialized already.')
        if issubclass(Detector, BaseDetector) is False:
            raise Exception('Given detector is not a subclass of Detector.')

        if isinstance(evaluator, BaseEvaluator) is False:
            raise Exception('Evaluator not initialized.')

        if model_ids is None:
            model_ids = self.model_ids
        elif len(list(set(model_ids).difference(set(self.model_ids)))) > 0:
            raise Exception('Unknown model ids.')

        scores = {model_id: [] for model_id in model_ids}
        processing_times = []

        for model_id in tqdm(model_ids,
                             total=len(model_ids),
                             disable=verbosity < 1,
                             desc='[' + self.id + '] Scoring Models...', ):
            detector = Detector(evaluator=evaluator, params=detector_params, config=detector_config, verbose=verbosity > 2)
            rtf_ids = self.get_rtf_ids_for_model(model_id)
            for rtf_id in tqdm(rtf_ids,
                               total=len(rtf_ids),
                               disable=verbosity < 2,
                               desc='[' + self.id + '] Scoring RTFs of Model ' + model_id + '...', ):
                try:
                    rtf = self.get_rtf(rtf_id)
                    df_score = rtf.score(detector)

                    processing_times = processing_times + df_score.processing_time.tolist()
                    evaluation = evaluator.evaluate_rtf(df_score)

                    if "evaluation" not in evaluation:
                        raise Exception('Dictionary must at least contain a value for "evaluation".')
                    if "id" in evaluation:
                        raise Exception('Key "id" is reserved.')

                    evaluation["id"] = rtf_id
                    scores[model_id].append(evaluation)
                except Exception as e:
                    raise Exception('Evaluation on RtF {} of dataset {} failed: {}'.format(rtf_id, self.id, e))

        scores_list = [scores[model_id] for model_id in scores]
        return {
            'final_score': evaluator.combine_rtf_evaluations([item['evaluation'] for sublist in scores_list for item in sublist]),
            'model_scores': [{'model_id': model_id, 'rtf_scores': scores[model_id]} for model_id in scores],
            'processing_times': {
                'min': min(processing_times),
                'max': max(processing_times),
                'avg': sum(processing_times, timedelta()) / len(processing_times)
            }
        }

    def evaluate_on_equipments(self,
                               Detector: Type[BaseDetector],
                               evaluator: BaseEvaluator,
                               model_id: str,
                               equipment_ids: [str] = None,
                               detector_params: dict = None,
                               detector_config: dict = None,
                               verbosity: int = 0):
        """

        :param Detector:
        :param evaluator:
        :param model_id:
        :param equipment_ids:
        :param detector_params:
        :param detector_config:
        :param verbosity:
        :return:
        """
        if isinstance(Detector, BaseDetector) is True:
            raise Exception('Detector is initialized already.')
        if issubclass(Detector, BaseDetector) is False:
            raise Exception('Given detector is not a subclass of Detector.')

        if isinstance(evaluator, BaseEvaluator) is False:
            raise Exception('Evaluator not initialized.')

        if equipment_ids is None:
            equipment_ids = self.get_equipment_ids_for_model(model_id=model_id)
        elif len(list(set(equipment_ids).difference(set(self.get_equipment_ids_for_model(model_id=model_id))))) > 0:
            raise Exception('Unknown model ids for model ' + str(model_id))

        scores = {equi_id: [] for equi_id in equipment_ids}
        processing_times = []

        detector = Detector(evaluator=evaluator, params=detector_params, config=detector_config, verbose=verbosity > 2)

        for equi_id in tqdm(equipment_ids,
                            total=len(equipment_ids),
                            disable=verbosity < 1,
                            desc='[' + self.id + '] Scoring Equipments...', ):
            rtf_ids = self.get_rtf_ids_for_equipment(model_id=model_id, equipment_id=equi_id)
            for rtf_id in tqdm(rtf_ids,
                               total=len(rtf_ids),
                               disable=verbosity < 2,
                               desc='[' + self.id + '] Scoring RTFs of Equipment ' + equi_id + '...', ):
                try:
                    rtf = self.get_rtf(rtf_id)
                    df_score = rtf.score(detector)

                    processing_times = processing_times + df_score.processing_time.tolist()
                    evaluation = evaluator.evaluate_rtf(df_score)

                    if "evaluation" not in evaluation:
                        raise Exception('Dictionary must at least contain a value for "evaluation".')
                    if "id" in evaluation:
                        raise Exception('Key "id" is reserved.')

                    evaluation["id"] = rtf_id
                    scores[equi_id].append(evaluation)
                except Exception as e:
                    raise Exception('Evaluation on RtF {} of dataset {} failed: {}'.format(rtf_id, self.id, e))

        scores_list = [scores[equi_id] for equi_id in scores]
        return {
            'final_score': evaluator.combine_rtf_evaluations([item['evaluation'] for sublist in scores_list for item in sublist]),
            'equipment_scores': [{'equi_id': equi_id, 'rtf_scores': scores[equi_id]} for equi_id in scores],
            'processing_times': {
                'min': min(processing_times),
                'max': max(processing_times),
                'avg': sum(processing_times, timedelta()) / len(processing_times)
            }
        }
