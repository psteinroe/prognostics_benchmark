import os
from pathlib import Path
import __main__
import logging
import importlib
from subprocess import Popen

from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from prognostics_benchmark.benchmark import Benchmark


class Optimizer:

    def __init__(
            self,
            out_path: str,
            p_bounds: dict,
            evaluator_name: str,
            detector_name: str,
            name: str = None,
            exact: bool = False,
            evaluator_config: dict = None,
            detector_config: dict = None,
            dataset_id: str = None,
            dataset_ids: [str] = None,
            model_ids: [str] = None,
            rtf_id: str = None,
            init_points: int = 5,
            n_iter: int = 15
    ):
        if name is not None:
            self.name = name
        else:
            self.name = __main__.__file__.replace('.py', '')
        self.exact = exact  # if true, checks that bounds contain all params

        Path(out_path).mkdir(parents=True, exist_ok=True)
        self.out_path = out_path

        self.benchmark = Benchmark()

        self.dataset_ids = dataset_ids
        self.dataset_id = dataset_id
        if self.dataset_id is not None:
            self.dataset = self.benchmark.get_dataset(dataset_id)
        self.model_ids = model_ids
        self.rtf_id = rtf_id

        self.evaluator_name = evaluator_name
        self.Evaluator = getattr(importlib.import_module("prognostics_benchmark.evaluators"), evaluator_name)
        self.evaluator = None
        if self.dataset_id is not None:
            self.evaluator = self.Evaluator(dataset_id=dataset_id, config=evaluator_config)

        self.detector_name = detector_name
        self.Detector = getattr(importlib.import_module("prognostics_benchmark.detectors"), detector_name)
        self.detector_config = detector_config
        self.detector_default_params = self.Detector.get_default_params()

        self.init_points = init_points
        self.n_iter = n_iter

        self.idx = 0
        self.max_iterations = self.init_points + self.n_iter

        self.p_bounds = p_bounds

        self.opt_filepath, self.error_filepath = self._find_log_paths()
        logging.basicConfig(
            handlers=[
                logging.FileHandler(self.error_filepath),
                logging.StreamHandler()
            ],
            level=logging.INFO
        )

    def _black_box_function(self, **kwargs):
        self.idx = self.idx + 1
        params = locals().get('kwargs')

        for key in params:
            if type(self.detector_default_params[key]) == int:
                params[key] = int(params[key])

        progress = "{}/{}".format(self.idx, self.max_iterations)

        # Fill up missing params
        for key in self.detector_default_params:
            if key not in params:
                if self.exact is True:
                    raise Exception('Bounds for {} not provided. Aborting.'.format(key))
                params[key] = self.detector_default_params[key]
                print('Bounds for {} not provided. Using default.'.format(key))

        score = 0
        individial_scores = []
        try:
            if self.rtf_id is not None:
                # Score RTF
                logging.info('#### ' + progress + ' - ' + self.rtf_id + ' ####')
                rtf = self.dataset.get_rtf(self.rtf_id)
                detector = self.Detector(evaluator=self.evaluator, params=params, config=self.detector_config)
                df_score = rtf.score(detector, verbose=True)
                rtf_evaluation = self.evaluator.evaluate_rtf(df_score)
                score = rtf_evaluation["evaluation"]
            elif self.model_ids is not None:
                # Score Models
                logging.info('#### ' + progress + ' - ' + ','.join(self.model_ids) + ' ####')
                res = self.dataset.evaluate_on_models(
                    model_ids=self.model_ids,
                    Detector=self.Detector,
                    evaluator=self.evaluator,
                    detector_params=params,
                    detector_config=self.detector_config,
                    verbosity=1)
                score = res.get('final_score')
                individial_scores = res.get('model_scores')
            elif self.dataset_id is not None:
                # Score dataset
                logging.info('#### ' + progress + ' - ' + self.dataset_id + ' ####')
                res = self.dataset.evaluate_on_models(
                    Detector=self.Detector,
                    evaluator=self.evaluator,
                    detector_params=params,
                    detector_config=self.detector_config,
                    verbosity=1)
                score = res.get('final_score')
                individial_scores = res.get('model_scores')
            else:
                # Score benchmark
                logging.info('#### ' + progress + ' - ' + 'Benchmark' + ' ####')
                res = self.benchmark.score(
                    detector_name=self.detector_name,
                    detector_params=params,
                    detector_config=self.detector_config,
                    evaluator_name=self.evaluator_name,
                    dataset_ids=self.dataset_ids,
                    verbose=True
                )
                score = res.get('final_score')
                individial_scores = {
                    dataset_score.get('dataset_id'): dataset_score.get('score')
                    for dataset_score in res.get('dataset_scores')
                }
        except Exception as e:
            logging.exception("Could not get score.")
            logging.exception(e)

        logging.info(params)
        logging.info('Score: ' + str(score))
        logging.info('Individual Scores:')
        logging.info(individial_scores)

        return score

    def _find_log_paths(self):
        base_name = self.name + '_'
        if self.rtf_id is not None:
            base_name = base_name + self.rtf_id + '_'
        elif self.dataset_id is not None:
            base_name = base_name + self.dataset_id + '_'

        idx = 0
        while os.path.isfile(os.path.join(self.out_path, base_name + str(idx) + '_opt.json')) | \
                os.path.isfile(os.path.join(self.out_path, base_name + str(idx) + '_logging.json')):
            idx = idx + 1
        return os.path.join(self.out_path, base_name + str(idx) + '_opt.json'), \
               os.path.join(self.out_path, base_name + str(idx) + '_logging.json')

    def optimize(self, load_from: str = None):
        self.idx = 0
        Popen('ulimit -n 4096', shell=True)

        optimizer = BayesianOptimization(
            f=self._black_box_function,
            pbounds=self.p_bounds,
            random_state=1,
        )

        logger = JSONLogger(path=self.opt_filepath)
        optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

        if load_from is not None:
            logfile = os.path.join(self.out_path, load_from)
            if Path(logfile).is_file():
                logging.info('Loading logs from ' + logfile)
                load_logs(optimizer, logs=[logfile])
            else:
                logging.info('Could not find a log file under {}'.format(logfile))

        optimizer.maximize(
            init_points=self.init_points,
            n_iter=self.n_iter,
        )
