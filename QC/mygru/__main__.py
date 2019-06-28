import sys
sys.path.extend(['.', '..', '../..'])

from QC.mygru.model import MyGRUModelQC

import torch
from typing import Tuple, Any
import argparse
from datetime import datetime
from hyperopt import hp
from hyperopt.pyll.base import scope

from common import GridCellMeasurements, PerformanceMetrics, count_parameters
from common.Datasets.trec import TRECDataset

import common.experiment
from common.experiment import ExtraArgs


argparser = argparse.ArgumentParser()
argparser.add_argument('--searches', type=int, default=60)
argparser.add_argument('--trials', type=int, default=3)
argparser.add_argument('--final-trials', type=int, default=1)
argparser.add_argument('--debug', action='store_true')
argparser.add_argument('--logname', type=str, default='')
argparser.add_argument('--epochs', type=int, help="Override the number of epochs")
argparser.add_argument('--saveto', type=str, default='')
args = argparser.parse_args()

train_fold, val_fold, test_fold = TRECDataset.splits(root=common.get_cache_path())

extra = ExtraArgs(
    ds={
        'train': train_fold,
        'validation': val_fold,
        'test': test_fold
    }
)

opt_space = {'reservoir_size': 90, 'lr': 0.01, 'epochs': 45, 'weight_decay': 1e-06, 'n_batch': 100}


def evaluate(hp: dict, extra: ExtraArgs, trial_id=0) -> Tuple[float, GridCellMeasurements]:
    """
    :return: a tuple composed of
      - a float specifying the score used for the random search (usually the validation accuracy)
      - a GridCellMeasurements object
    """
    if trial_id == 0:
        print(hp)

    train = extra.ds['train']
    val = extra.ds['validation']
    test = None

    if extra.is_final_trials:
        train = TRECDataset.merge_folds([extra.ds['train'], val])
        val = None
        test = extra.ds['test']

    model = MyGRUModelQC(hp, logname=args.logname)

    if trial_id == 0:
        print(f"# parameters: {count_parameters(model)}")

    model.fit(train, val)

    train_perf, val_perf, test_perf = model.performance(train, val, test)

    if extra.is_final_trials:
        # Save the model
        datet = datetime.now().strftime('%b%d_%H-%M-%S')
        filename = f'QC_mygru_{datet}_{trial_id}_{round(test_perf*100, 1)}.pt'
        torch.save(model.state_dict(), filename)

    metric_type = PerformanceMetrics.accuracy

    measurements = GridCellMeasurements(
        train_perf=train_perf,
        val_perf=val_perf,
        test_perf=test_perf,
        metric=metric_type.name,
        training_time=model.training_time,
        extra={
            metric_type.name: {
                'train': train_perf,
                'val': val_perf,
                'test': test_perf,
            }
        }
    )

    loss = 1/val_perf if val_perf > 0 else float('inf')
    return loss, measurements


if args.epochs is not None:
    opt_space['epochs'] = args.epochs

common.experiment.run_experiment(
    opt_space,
    evaluate,
    args.searches,
    args.trials,
    args.final_trials,
    extra
)
