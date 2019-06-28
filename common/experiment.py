from common.randsearch import execute_trials, EvaluationFunction, ModelSelectionResult, print_final_trials_result
import hyperopt
from typing import Any
from dataclasses import dataclass


@dataclass
class ExtraArgs:
    ds: Any = None
    log_path: str = None
    is_final_trials: bool = False


def run_experiment(opt_space,
                   evaluator: EvaluationFunction,
                   n_searches: int,
                   n_trials: int,
                   n_final_trials: int,
                   extra: ExtraArgs):

    def opt_wrapper(hh):
        perf, measurements = execute_trials(hh, evaluator, n_trials, extra=extra)
        return {'loss': perf, 'status': hyperopt.STATUS_OK, 'user_data': measurements, 'hp': hh}

    print("Hyperparameters grid:")
    print(opt_space)
    print("")

    extra.is_final_trials = False

    if n_searches > 0:
        trials = hyperopt.Trials()
        best = hyperopt.fmin(
            opt_wrapper,
            space=opt_space,
            algo=hyperopt.partial(hyperopt.mix.suggest, p_suggest=[
                (.5, hyperopt.rand.suggest),
                (.4, hyperopt.tpe.suggest),
                (.1, hyperopt.anneal.suggest)]),
            max_evals=n_searches,
            trials=trials
        )

        print(trials.best_trial)
        print(best)

        best_hyperparams = trials.best_trial['result']['hp']
        msr = ModelSelectionResult(opt_space, best_hyperparams, n_searches, n_trials,
                                   trials.best_trial['result']['user_data'])
    else:
        best_hyperparams = opt_space
        msr = ModelSelectionResult(opt_space, best_hyperparams, n_searches, n_trials, [])

    print(f"\n\nNow running {n_final_trials} final trials...")

    # Enable the computation of the scores on the test set
    extra.is_final_trials = True

    # Compute more accurate values for the scores associated to the best hyperparametrization
    _, final_measures = execute_trials(best_hyperparams, evaluator, n_final_trials, extra=extra)

    print_final_trials_result(opt_space, msr, final_measures)

