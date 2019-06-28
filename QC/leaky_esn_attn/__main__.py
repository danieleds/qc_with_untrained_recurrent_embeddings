import sys
sys.path.extend(['.', '..', '../..'])

from QC.leaky_esn_attn.model import ESNModelQC

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

opt_space = {
    'reservoir_size': 500 * scope.int(hp.quniform('reservoir_size', 1, 10, q=1)),
    'num_layers': 1,
    'density_in': [hp.uniform('density_in', 0.001, 1)],
    'density_in_bw': [hp.uniform('density_in_bw', 0.001, 1)],
    'scale_in': [hp.loguniform('scale_in', -7, 4)],
    'scale_in_bw': [hp.loguniform('scale_in_bw', -7, 4)],
    'scale_rec': [hp.loguniform('scale_rec', -7, 4)],
    'scale_rec_bw': [hp.loguniform('scale_rec_bw', -7, 4)],
    'leaking_rate': [hp.uniform('leaking_rate', 0, 1)],
    'leaking_rate_bw': [hp.uniform('leaking_rate_bw', 0, 1)],
    'dropout': hp.uniform('dropout', 0.001, 1),
    'lr': hp.loguniform('lr', -9, -3),
    'n_attention': 2**scope.int(hp.quniform('n_attention', 7, 9, q=1)),
    'attention_r': 1, #hp.quniform('attention_r', 1, 8, q=1),
    'mlp_n_hidden': 0,  # ['int_uniform', 0, 10],
    'mlp_hidden_size': 0,  # ['choice', [128, 256, 512, 1024, 2048]],
    'epochs': 500,
    'n_batch': 128,  # ['choice', [64, 96, 128]]
    'weight_decay': hp.loguniform('weight_decay', -9, 0),
    'attention_type': 'LinSelfAttention'  # 'LinSelfAttention', 'Attention', 'MaxPooling', 'Mean', 'None'
}

opt_space = {'attention_r': 1, 'attention_type': 'LinSelfAttention', 'density_in': (0.6490642394189948,), 'density_in_bw': (0.6745601715844801,), 'dropout': 0.7970124251550051, 'epochs': 60, 'leaking_rate': (0.38930217546173795,), 'leaking_rate_bw': (0.7738524701335754,), 'lr': 0.000138326471408694, 'mlp_hidden_size': 0, 'mlp_n_hidden': 0, 'n_attention': 256, 'n_batch': 128, 'num_layers': 1, 'reservoir_size': 3000, 'scale_in': (6.745137376597286,), 'scale_in_bw': (0.2725147494411861,), 'scale_rec': (0.04641425166402264,), 'scale_rec_bw': (0.49182066901748145,), 'weight_decay': 0.0012623295223805023}


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

    model = ESNModelQC(hp, logname=args.logname)

    if trial_id == 0:
        print(f"# parameters: {count_parameters(model.model)}")

    model.fit(train, val)

    train_perf, val_perf, test_perf = model.performance(train, val, test)

    if extra.is_final_trials:
        # Save the model
        datet = datetime.now().strftime('%b%d_%H-%M-%S')
        filename = f'QC_leaky-esn-attn_{datet}_{trial_id}_{round(test_perf*100, 1)}.pt'
        torch.save(model.model.state_dict(), filename)

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
