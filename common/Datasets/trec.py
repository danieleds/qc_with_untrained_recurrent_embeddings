import torch
from torch.utils.data import Dataset
from typing import List, Tuple
import itertools
import os
import torchtext


class TRECDataset(Dataset):

    # http://cogcomp.org/Data/QA/QC/
    # http://cogcomp.org/Data/QA/QC/train_5500.label
    # With FastText embeddings already applied.

    url = 'https://github.com/danieleds/qc_with_untrained_recurrent_embeddings/releases/download/v0.1/question_classification.bin'

    def __init__(self, examples: List[Tuple[torch.Tensor, torch.ByteTensor]]):
        super(TRECDataset, self).__init__()
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]

        return {
            'x': e[0].type(torch.get_default_dtype()),
            'y': e[1].argmax(dim=0).unsqueeze(0)
        }

    @classmethod
    def load(cls, root='.data'):
        ds_path = os.path.join(root, 'TREC-QC', 'question_classification_ft.bin')
        if not os.path.isfile(ds_path):
            if not os.path.exists(os.path.dirname(ds_path)):
                os.makedirs(os.path.dirname(ds_path))
            print('downloading dataset')
            torchtext.utils.download_from_url(TRECDataset.url, ds_path)
        return torch.load(ds_path)

    @classmethod
    def splits(cls, root='.data'):
        """
        Returns a training set, a validation set, a test set.
        :return: (training set, validation set, test set)
        """

        ds = cls.load(root=root)

        train_fold = ds['folds']['train']
        val_fold = ds['folds']['validation']
        test_fold = ds['folds']['test']

        train = [(train_fold['input'][i], train_fold['target'][i]) for i in range(len(train_fold['target']))]
        val = [(val_fold['input'][i], val_fold['target'][i]) for i in range(len(val_fold['target']))]
        test = [(test_fold['input'][i], test_fold['target'][i]) for i in range(len(test_fold['target']))]

        train_ds = cls(train)
        val_ds = cls(val)
        test_ds = cls(test)

        return train_ds, val_ds, test_ds

    @classmethod
    def merge_folds(cls, folds: List['TRECDataset']):
        examples = list(itertools.chain.from_iterable([ f.examples for f in folds ]))
        return cls(examples)
