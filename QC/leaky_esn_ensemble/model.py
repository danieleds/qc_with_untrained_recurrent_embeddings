import torch
from QC.leaky_esn.model import ESNModelQC
import common

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ESNModelQCEnsemble(torch.nn.Module):

    def __init__(self, n_models, input_size, reservoir_size, f=torch.tanh, alpha=1e-6,
                 rescaling_method='norm', hp=None):
        super(ESNModelQCEnsemble, self).__init__()

        self.models = [ ESNModelQC(
            input_size,
            reservoir_size,
            f=f,
            alpha=alpha,
            rescaling_method=rescaling_method,
            hp=hp
        ) for _ in range(n_models) ]

        self.batch_size = hp['n_batch']

        self.training_time = -1

    def find_best_alpha(self, train_fold, val_fold):
        """
        Fit the model while searching for the best regularization parameter. The best regularization
        parameter is then assigned to self.alpha.
        :param train_fold:
        :param val_fold:
        :param batch_size:
        :return:
        """

        self.models[0].find_best_alpha(train_fold, val_fold, self.batch_size)
        best_alpha = self.models[0].alpha

        for m in self.models:
            m.alpha = best_alpha

        return best_alpha

    def fit(self, train_fold):
        """
        Fits the model with self.alpha as regularization parameter.
        :param train_fold: training fold.
        :param batch_size:
        :return:
        """

        for m in self.models:
            m.fit(train_fold)

        self.training_time = sum([m.training_time for m in self.models])

    def performance(self, train_fold, val_fold, test_fold=None):
        with torch.no_grad():

            # Expected output class indices
            train_expected = torch.Tensor([ d['y'] for d in train_fold ])
            val_expected = torch.Tensor([ d['y'] for d in val_fold ]) if val_fold else None
            test_expected = torch.Tensor([ d['y'] for d in test_fold ]) if test_fold else None

            for m in self.models:
                m.eval()

            train_outs = [ m.forward_in_batches(train_fold, self.batch_size, return_probs=True) for m in self.models ]
            val_outs = [ m.forward_in_batches(val_fold, self.batch_size, return_probs=True) for m in self.models ] if val_fold else None
            test_outs = [ m.forward_in_batches(test_fold, self.batch_size, return_probs=True) for m in self.models ] if test_fold else None

            train_out = torch.stack(train_outs).mean(dim=0).argmax(dim=1)
            val_out = torch.stack(val_outs).mean(dim=0).argmax(dim=1) if val_fold else None
            test_out = torch.stack(test_outs).mean(dim=0).argmax(dim=1) if test_fold else None

            # Compute performance measures
            train_accuracy = common.accuracy(train_out, train_expected)
            val_accuracy = common.accuracy(val_out, val_expected) if val_fold else 0
            test_accuracy = common.accuracy(test_out, test_expected) if test_fold else 0

            return train_accuracy, val_accuracy, test_accuracy
