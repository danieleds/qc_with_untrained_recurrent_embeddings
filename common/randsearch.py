import numpy as np
import datetime
from typing import Callable, Any, Tuple, List
from dataclasses import dataclass

from common import GridCellMeasurements

EvaluationFunction = Callable[[dict, Any, int], Tuple[float, GridCellMeasurements]]


@dataclass
class ModelSelectionResult:
    """
    Result of a model selection (e.g. a random search).

    Attributes:
        search_space    the hyperparameters search space used for this model selection
        hyperparams     the best hyperparameters that have been selected
        n_searches      the number of explored hyperparametrizations
        n_trials        the number of trials for each parametrization
        trials_measures             trials_measures information associated to the best model. This is an array with
                        one entry for each trial.
    """
    search_space: dict
    hyperparams: dict
    n_searches: int
    n_trials: int
    trials_measures: List[GridCellMeasurements]


def execute_trials(hyperparameters: dict, evaluation_fn: EvaluationFunction, trials: int, extra=None) \
        -> Tuple[float, List[GridCellMeasurements]]:
    """
    Executes a given number of trials for a specific hyperparametrization
    :param hyperparameters: a set of hyperparameters. Entries can be written by evaluation_fn!
    :param evaluation_fn:
    :param trials:
    :param extra:
    :return: a tuple:
        - scalar float value representing the average performance over the trials (it is used for ranking this
          parametrization: its absolute value is not important)
        - list of GridCellMeasurements, one for each trial.
    """
    # Since evaluation_fn can modify the current hyperparameters, the order of the trials must
    # be preserved!
    evaluations = [evaluation_fn(hyperparameters, extra, t) for t in range(trials)]

    # Compute the average performance for the 'validation' score over the trials
    avg_perf = float(np.mean([ev[0] for ev in evaluations]))

    # Save the raw information from each evaluation (contains training, validation and test scores)
    raw_scores = [ev[1] for ev in evaluations]

    return avg_perf, raw_scores


def mean_std_performances(data: List[GridCellMeasurements]) -> dict:
    """
    Given a list of metrics (one for each trial) compute mean and std dev for each measure.
    :param data:
    :return:
    """
    return {
        'train_avg': np.mean([d.train_perf for d in data]),
        'val_avg': np.mean([d.val_perf for d in data]),
        'test_avg': np.mean([d.test_perf for d in data]),
        'train_std': np.std([d.train_perf for d in data]),
        'val_std': np.std([d.val_perf for d in data]),
        'test_std': np.std([d.test_perf for d in data]),
        'training_time_avg': np.mean([d.training_time for d in data]),
        'training_time_std': np.std([d.training_time for d in data]),
    }


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def print_final_trials_result(hyperparams_search_space, best: ModelSelectionResult,
                              test_measurements: List[GridCellMeasurements]):
    best_measures = mean_std_performances(best.trials_measures)
    best_test_measures = mean_std_performances(test_measurements)

    lines = [
        "",
        f"Best result:     train {best_measures['train_avg']:.6f}  val {best_measures['val_avg']:.6f}  test {best_test_measures['test_avg']:.6f}",
        f"                 (std) {best_measures['train_std']:.6f}      {best_measures['val_std']:.6f}       {best_test_measures['test_std']:.6f}",
        "",
        f"Training time:         {best_test_measures['training_time_avg']:.6f} sec.  ({hms_string(best_test_measures['training_time_avg'])})",
        f"                 (std) {best_test_measures['training_time_std']:.6f}",
        "",
        "Best hyperparameters:",
        str(best.hyperparams),
        "",
        "Search space:",
        str(hyperparams_search_space),
        "",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ""
    ]

    for line in lines:
        print(line)
