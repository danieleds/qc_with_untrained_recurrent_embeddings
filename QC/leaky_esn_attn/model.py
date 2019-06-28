import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

import common
import common.embeddings
from common.custom_models.leaky_esn_attention import LeakyESNAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_fn(samples):
    return {
        'x': torch.nn.utils.rnn.pad_sequence([ d['x'] for d in samples ], batch_first=False),
        'lengths': [ len(d['x']) for d in samples ],
        'y': torch.stack([d['y'] for d in samples])
    }


def get_log_dir(comment=""):
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'runs', current_time + '_' + comment)
    return os.path.join(os.path.realpath(__file__ + "/../"), log_dir)


class ESNModelQC:

    def __init__(self, hp, logname=''):

        self.model = LeakyESNAttention(
            input_size=300,
            output_size=6,
            reservoir_size=hp['reservoir_size'],
            num_esn_layers=hp['num_layers'],
            mlp_n_hidden=hp['mlp_n_hidden'],
            mlp_hidden_size=hp['mlp_hidden_size'],
            dropout=hp['dropout'],
            attention_type=hp['attention_type'],
            attention_hidden_size=hp['n_attention'],
            attention_heads=hp['attention_r'],
            scale_rec=hp['scale_rec'],
            scale_rec_bw=hp['scale_rec_bw'],
            scale_in=hp['scale_in'],
            scale_in_bw=hp['scale_in_bw'],
            density_in=hp['density_in'],
            density_in_bw=hp['density_in_bw'],
            leaking_rate=hp['leaking_rate'],
            leaking_rate_bw=hp['leaking_rate_bw']
        ).to(device)

        self.epochs = hp['epochs']
        self.lr = hp['lr']
        self.batch_size = hp['n_batch']
        self.weight_decay = hp['weight_decay']

        self.training_time = -1

        self.log_writer = SummaryWriter(log_dir=get_log_dir(comment=logname))
        self.log_writer.add_text("hp", str(hp))

    def forward_in_batches(self, dataset, batch_size):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)

        _Xs = []
        for _, minibatch in enumerate(dataloader):
            data_x = minibatch['x'].to(device, non_blocking=True)
            _Xs += [self.model.forward(data_x, seq_lengths=minibatch['lengths'])]

        return torch.cat(_Xs, dim=0)

    def fit(self, train_fold, val_fold):
        """
        Fits the model with self.alpha as regularization parameter.
        :param train_fold: training fold.
        :param val_fold: validation fold.
        :return:
        """

        t_train_start = time.time()

        epochs = self.epochs

        #weights = 1.0 / torch.Tensor([67, 945, 1004, 949, 670, 727])
        #sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_fold))
        #dataloader = DataLoader(train_fold, batch_size=self.batch_size, collate_fn=collate_fn,
        #                        pin_memory=True, sampler=sampler)

        dataloader = DataLoader(train_fold, shuffle=True, batch_size=self.batch_size, collate_fn=collate_fn, pin_memory=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        criterion = torch.nn.CrossEntropyLoss()

        best_loss = float('inf')
        epochs_without_improvement = 0
        best_val_accuracy = 0
        epochs_without_val_acc_improvement = 0
        for epoch in tqdm(range(1, epochs + 1), desc="epochs", dynamic_ncols=True):
            running_loss = 0.0
            num_minibatches = 0
            for i, data in enumerate(dataloader):
                # Move data to devices
                data_x = data['x'].to(device, non_blocking=True)
                data_y = data['y'].to(device, non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                self.model.train()

                train_out = self.model.forward(data_x, seq_lengths=data['lengths'])
                train_expected = data_y.squeeze(dim=1)

                loss = criterion(train_out, train_expected) + self.model.loss_penalty()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num_minibatches += 1

            curr_avg_loss = running_loss/num_minibatches

            if curr_avg_loss < best_loss:
                epochs_without_improvement = 0
                best_loss = curr_avg_loss
            else:
                epochs_without_improvement += 1

            self.log_writer.add_scalar(
                tag="data/training_loss",
                scalar_value=curr_avg_loss,
                global_step=epoch)

            skip = 10
            if epoch % skip == 0 and val_fold is not None:
                _, val_accuracy, _ = self.performance(None, val_fold, None)
                self.log_writer.add_scalar(
                    tag="data/validation_accuracy_curve",
                    scalar_value=val_accuracy,
                    global_step=epoch)

                if val_accuracy > best_val_accuracy:
                    epochs_without_val_acc_improvement = 0
                    best_val_accuracy = val_accuracy
                else:
                    epochs_without_val_acc_improvement += skip

                # Early stopping
                if epochs_without_val_acc_improvement >= 10:
                    print(f"Epoch {epoch}: no accuracy improvement after 10 epochs. Early stop.")
                    # FIXME Remember for final_trials!!
                    break

            ## Early stopping
            #if epochs_without_improvement >= 10:
            #    print(f"Epoch {epoch}: no loss improvement after 10 epochs. Early stop.")
            #    break

        t_train_end = time.time()
        self.training_time = t_train_end - t_train_start
        self.log_writer.close()

    def performance(self, train_fold, val_fold, test_fold=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if train_fold:
            train_accuracy, train_out, train_expected = self.performance_from_fold(train_fold, batch_size)
        else:
            train_accuracy, train_out, train_expected = (0, None, None)

        if val_fold:
            val_accuracy, val_out, val_expected = self.performance_from_fold(val_fold, batch_size)
            self.log_writer.add_scalar("data/validation_accuracy", val_accuracy)
        else:
            val_accuracy, val_out, val_expected = (0, None, None)

        if test_fold:
            test_accuracy, test_out, test_expected = self.performance_from_fold(test_fold, batch_size)
            self.log_writer.add_scalar("data/test_accuracy", test_accuracy)
        else:
            test_accuracy, test_out, test_expected = (0, None, None)

        save_raw_predictions = False
        if save_raw_predictions:
            raw_preds_filename = '/home/disarli/tmp/predictions.pt'
            try:
                saved = torch.load(raw_preds_filename)
            except FileNotFoundError:
                saved = []
            saved.append({
                'train_out': train_out.cpu(),
                'train_expected': train_expected.cpu(),
                'val_out': val_out.cpu(),
                'val_expected': val_expected.cpu(),
                'test_out': test_out.cpu() if test_fold else None,
                'test_expected': test_expected.cpu() if test_fold else None,
            })
            torch.save(saved, raw_preds_filename)

        return train_accuracy, val_accuracy, test_accuracy

    def performance_from_out(self, output, expected):
        """
        Given a tensor of network outputs and a tensor of expected outputs, returns the performance
        :param output:
        :param expected:
        :return:
        """
        output = output.argmax(dim=1).cpu()

        return common.accuracy(output, expected)

    def performance_from_fold(self, fold, batch_size):
        with torch.no_grad():
            self.model.eval()

            out = self.forward_in_batches(fold, batch_size)
            expected = torch.Tensor([d['y'] for d in fold])

            perf = self.performance_from_out(out, expected)
            return perf, out, expected
