import os
from pathlib import Path
from datetime import timedelta
import pandas as pd
import shutil

from prognostics_benchmark import Benchmark

if __name__ == '__main__':
    benchmark = Benchmark()

    results_path = Path('results')
    if results_path.exists() and results_path.is_dir():
        print('Removing old results.')
        shutil.rmtree(results_path)

    for evaluator_name in benchmark.get_evaluator_names():
        evaluator_summary = []  # contains scores of all detectors using this evaluator --> results/summary_{}.csv

        for detector_name in benchmark.get_detector_names():
            print('Scoring {} with {}...'.format(detector_name, evaluator_name))
            res = None
            while res is None:
                try:
                    res = benchmark.score(
                        detector_name=detector_name,
                        evaluator_name=evaluator_name,
                        verbose=True
                    )
                except Exception as e:
                    print("Error for Detector {}. {}. Skipping...".format(detector_name, e))
            print('Done. Parsing and Writing Results...')
            summary_dict = {
                'detector_name': detector_name,
                'final_score': res.get('final_score'),
                **{dataset_score.get('dataset_id'): dataset_score.get('score') for dataset_score in res.get('dataset_scores')},
                **{"{}_processing_time_ms".format(key): val / timedelta(milliseconds=1) for key, val in res.get('processing_times').items() if key != 'min'}
            }
            evaluator_summary.append(summary_dict)

            dataset_summaries = []
            for dataset_score in res.get('dataset_scores'):
                dataset_summary = {
                    'dataset_id': dataset_score.get('dataset_id'),
                    'score': dataset_score.get('score'),
                    **{"{}_processing_time_ms".format(key): val / timedelta(milliseconds=1) for key, val in
                       res.get('processing_times').items()}
                }
                dataset_summaries.append(dataset_summary)
                model_scores = {model_score.get('model_id'): model_score.get('rtf_scores') for model_score in dataset_score.get('model_scores')}

                model_summaries_out_path = os.path.join(results_path, detector_name, evaluator_name, dataset_score.get('dataset_id'))
                Path(model_summaries_out_path).mkdir(parents=True, exist_ok=True)
                for model_id, scores in model_scores.items():
                    df_model_summary = pd.DataFrame(scores)
                    df_model_summary.to_csv(os.path.join(model_summaries_out_path, '{}.csv'.format(model_id)), index=False)
            df_dataset_summary = pd.DataFrame(dataset_summaries)
            evaluator_detector_summary_out_path = os.path.join(results_path, detector_name, evaluator_name)
            Path(evaluator_detector_summary_out_path).mkdir(parents=True, exist_ok=True)
            df_dataset_summary.to_csv(os.path.join(evaluator_detector_summary_out_path, 'summary.csv'), index=False)

        df_evaluator_summary = pd.DataFrame(evaluator_summary)
        Path(results_path).mkdir(parents=True, exist_ok=True)
        df_evaluator_summary.to_csv(os.path.join('results', './{}_results.csv'.format(evaluator_name)), index=False)


