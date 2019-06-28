import torch
import math
from enum import Enum
from dataclasses import dataclass
from sklearn.metrics import f1_score
import configparser
import os


_config = None


@dataclass
class GridCellMeasurements:
    train_perf: float = None
    val_perf: float = None
    test_perf: float = None
    metric: str = None
    training_time: float = None
    extra: dict = None

    def to_dict(self):
        return {
            'train_perf': self.train_perf,
            'val_perf': self.val_perf,
            'test_perf': self.test_perf,
            'metric': self.metric,
            'training_time': self.training_time,
            'extra': self.extra
        }

    def __str__(self):
        return str(self.to_dict())


class PerformanceMetrics(Enum):
    accuracy = 'accuracy'
    perplexity = 'perplexity'
    cross_entropy = 'cross_entropy'

    def lower_is_better(self):
        return self in [PerformanceMetrics.perplexity, PerformanceMetrics.cross_entropy]

    def higher_is_better(self):
        return not self.lower_is_better()


def accuracy(predicted: torch.Tensor, expected: torch.Tensor) -> float:
    """

    :param predicted: (n) tensor of predicted class indices, e.g. [ 2, ... ]
    :param expected: (n) tensor of expected class indices, e.g. [ 2, ... ]
    :return:
    """
    assert predicted.shape == expected.shape
    return (predicted.type(torch.long) == expected.type(torch.long)).sum().item() / len(expected)


def macro_f1_score(predicted: torch.Tensor, expected: torch.Tensor) -> float:
    """

    :param predicted: (n) tensor of predicted class indices, e.g. [ 2, ... ]
    :param expected: (n) tensor of expected class indices, e.g. [ 2, ... ]
    :return:
    """
    return f1_score(expected.cpu().numpy(), predicted.cpu().numpy(), average='macro')


def perplexity_softmax(predicted_probs: torch.Tensor, expected: torch.Tensor) -> float:
    """
    USE perplexity INSTEAD!
    :param predicted_probs: model predictions, WITH preapplied softmax
    :param expected:
    :return:
    """
    # Select the probability associated to the correct prediction, for each sample
    # (num_samples,)
    N = predicted_probs.shape[0]
    p = predicted_probs[torch.arange(N), expected.to(torch.long)]

    # Remove zeros
    # p[p < 1e-12] = 1e-12

    return (2 ** ((-1/N) * torch.sum(torch.log2(p)))).item()


def perplexity(predicted_probs: torch.Tensor, expected: torch.LongTensor) -> float:
    """
    Let p be the vector of probabilities (from softmax(predicted_probs)) associated to the expected classes.
    Then we have
            Perplexity = e ** ((-1/N) * sum(log_e(p)))
    But since
            cross_entropy = (-1/N) * sum(log_e(p)),
    we have that
            Perplexity = e ** cross_entropy

    In case the torch cross_entropy was computed with a different logarithm, e.g. log_2, we would have
    to use 2 ** cross_entropy instead.
    :param predicted_probs: model predictions, WITHOUT applying softmax
    :param expected:
    :return:
    """

    # predicted_probs = torch.softmax(predicted_probs, dim=1)
    # # Select the probability associated to the correct prediction, for each sample
    # # (num_samples,)
    # N = predicted_probs.shape[0]
    # p = predicted_probs[torch.arange(N), expected.to(torch.long)]
    # # Remove zeros
    # # p[p < 1e-12] = 1e-12
    # return (2 ** ((-1/N) * torch.sum(torch.log2(p)))).item()

    return math.exp(torch.nn.functional.cross_entropy(predicted_probs, expected))


def count_parameters(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_root_path():
    return os.path.realpath(__file__ + "/../../")


def get_cache_path():
    global _config
    if _config is None:
        _config = configparser.ConfigParser()
        _config.read(os.path.join(get_root_path(), 'config.ini'))
    cfg_dir = _config['DEFAULT']['cache_dir']
    if cfg_dir == '':
        return os.path.join(get_root_path(), '.cache')
    else:
        return cfg_dir
