import inspect
import importlib
from datetime import timedelta
from tqdm import tqdm
import statistics
from typing import Type
from multiprocessing import Pool, cpu_count
from prognostics_benchmark.datasets import DataManager
from prognostics_benchmark.dataset import Dataset
import prognostics_benchmark.detectors as detectors
import prognostics_benchmark.evaluators as evaluators
from prognostics_benchmark.detectors.base import Detector as BaseDetector
from prognostics_benchmark.detectors.base import Evaluator as BaseEvaluator


class Benchmark:
    def __init__(self):
        self.data_manager: DataManager = DataManager()

    @property
    def dataset_ids(self) -> [str]:
        return self.data_manager.dataset_ids

    def get_dataset(self, dataset_id: str) -> Dataset:
        if dataset_id not in self.dataset_ids:
            raise Exception('Unknown dataset ID.')

        return Dataset(dataset_id)

    @staticmethod
    def get_detector_names() -> [str]:
        return [m[0] for m in inspect.getmembers(detectors, inspect.isclass)]

    @staticmethod
    def get_evaluator_names() -> [str]:
        return [m[0] for m in inspect.getmembers(evaluators, inspect.isclass)]

    def score(self,
              detector_name: str,
              evaluator_name: str,
              dataset_ids: [str] = None,
              detector_params: dict = None,
              detector_config: dict = None,
              evaluator_config: dict = None,
              verbose: bool = False):
        if detector_name not in self.get_detector_names():
            raise Exception('Invalid detector name')
        if evaluator_name not in self.get_evaluator_names():
            raise Exception('Invalid evaluator name')

        if dataset_ids is None:
            dataset_ids = self.dataset_ids
        elif len(list(set(dataset_ids).difference(set(self.dataset_ids)))) > 0:
            raise Exception('Unknown dataset ids')

        # Check if scoring config is available for all data in dataset_ids
        Evaluator = getattr(importlib.import_module("prognostics_benchmark.evaluators"), evaluator_name)
        if len(list(set(dataset_ids).difference(set(Evaluator.get_default_config().keys())))) > 0:
            raise Exception('Scoring function does not have a default config for all required dataset ids')

        Detector = getattr(importlib.import_module("prognostics_benchmark.detectors"), detector_name)

        tasks = [
            (
                dataset_id,
                Detector,
                detector_params,
                detector_config,
                Evaluator(dataset_id=dataset_id, config=evaluator_config),
            ) for dataset_id in dataset_ids
        ]

        p = Pool(min(cpu_count(), len(tasks)))
        results = [
            score for score in tqdm(p.imap_unordered(_score_on_dataset_wrapper, tasks),
                                    total=len(dataset_ids),
                                    disable=not verbose,
                                    desc='Scoring {} with {}...'.format(detector_name, evaluator_name))
        ]

        scores = [r.get('score') for r in results]
        processing_times = [r.get('processing_times') for r in results]
        avg_processing_times = [times.get('avg') for times in processing_times]

        return {
            'final_score': Evaluator.combine_dataset_evaluations(scores),
            'dataset_scores': results,
            'processing_times': {
                'min': min([times.get('min') for times in processing_times]),
                'max': max([times.get('max') for times in processing_times]),
                'avg': sum(avg_processing_times, timedelta()) / len(avg_processing_times)
            }
        }


def _score_on_dataset_wrapper(args):
    return _score_on_dataset(*args)


def _score_on_dataset(dataset_id: str,
                      Detector: Type[BaseDetector],
                      detector_params: dict,
                      detector_config: dict,
                      evaluator: BaseEvaluator):
    dataset = Dataset(dataset_id)
    dataset_score = dataset.evaluate_on_models(
        Detector=Detector,
        evaluator=evaluator,
        detector_params=detector_params,
        detector_config=detector_config,
        verbosity=1
    )

    return {
        'dataset_id': dataset_id,
        'score': dataset_score['final_score'],
        'processing_times': dataset_score['processing_times'],
        'model_scores': dataset_score['model_scores']
    }
