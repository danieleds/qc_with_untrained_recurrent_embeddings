import torch
import torch.nn.functional as F
import warnings
import numbers
import math
from typing import Callable


class CustomRNNBase(torch.nn.Module):

    def __init__(self, cell_provider: Callable[[int, int], torch.nn.Module],
                 input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False):
        super(CustomRNNBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        num_directions = 2 if bidirectional else 1

        if batch_first:
            ValueError("batch_first=True is not supported yet")

        if not isinstance(dropout, numbers.Number) or not 0 <= dropout <= 1 or \
                isinstance(dropout, bool):
            raise ValueError("dropout should be a number in range [0, 1] "
                             "representing the probability of an element being "
                             "zeroed")
        if dropout > 0 and num_layers == 1:
            warnings.warn("dropout option adds dropout after all but last "
                          "recurrent layer, so non-zero dropout expects "
                          "num_layers greater than 1, but got dropout={} and "
                          "num_layers={}".format(dropout, num_layers))

        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else hidden_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                # Instantiate a cell and save it as an instance on this RNN
                cell = cell_provider(layer_input_size, hidden_size)
                setattr(self, 'cell_l{}{}'.format(layer, suffix), cell)

    def forward(self, input: torch.Tensor, h_0=None, seq_lengths=None):
        """
        input: (seq_len, batch, input_size)
        h_0: (num_layers * num_directions, batch, hidden_size)

        Outputs: output, h_n

        output: (seq_len, batch, num_directions * hidden_size)  # Only the output of the last layer, for both directions
                                                                  concatenated
        h_n: (num_layers * num_directions, batch, hidden_size)  # Hidden state for the last time step, in all layers.
            the layers can be separated using
                ``h_n.view(num_layers, num_directions, batch, hidden_size)``.

        """

        batch = input.size(1)
        num_directions = 2 if self.bidirectional else 1

        if h_0 is None:
            h_0 = input.new_zeros((self.num_layers * num_directions, batch, self.hidden_size))

        h_l = [h_0[i] for i in range(self.num_layers * num_directions)]

        last_layer_output = None  # (seq_len, batch, num_directions * self.hidden_size)
        for l in range(self.num_layers):
            layer_input = input if l == 0 else last_layer_output

            if l > 0 and self.dropout > 0:
                layer_input = F.dropout(layer_input, p=self.dropout)

            forward_outputs = []  # list of (batch, hidden_size)
            backward_outputs = []  # list of (batch, hidden_size)

            # Forward pass
            for step in layer_input:
                cell = getattr(self, 'cell_l{}'.format(l))
                input_state = h_l[num_directions * l + 0]
                h, c = cell.forward(step, input_state)  # (batch, hidden_size)
                h_l[num_directions * l + 0] = h
                forward_outputs = forward_outputs + [h]

            if self.bidirectional:
                # Backward pass
                cell = getattr(self, 'cell_l{}_reverse'.format(l))
                for step in reversed(layer_input):
                    input_state = h_l[num_directions * l + 1]
                    h, c = cell.forward(step, input_state)  # (batch, hidden_size)
                    h_l[num_directions * l + 1] = h
                    backward_outputs = [h] + backward_outputs

            if self.bidirectional:
                prev_output_forward = torch.stack(forward_outputs)  # (seq_len, batch, hidden_size)
                prev_output_backward = torch.stack(backward_outputs)  # (seq_len, batch, hidden_size)
                last_layer_output = torch.cat((prev_output_forward, prev_output_backward), dim=2)
            else:
                last_layer_output = torch.stack(forward_outputs)

        return last_layer_output, h_l

    def extract_last_time_step(self, states, seq_lengths=None):
        """
        Given the output of the GResNet for the whole sequence of a batch,
        extracts the values associated to the last time step.
        When seq_lengths is provided, this method takes particular care in ignoring the time steps
        associated with padding inputs, also in the case of a bidirectional architecture. For example,
        assume "P" indicates any state computed from a padding input, f1, ... fn are states computed in the
        forward pass, and b1, ... bn are states computed in the backward pass. Then, this method transforms
        the concatenated bidirectional output

                [ f1 f2 ... fn P P P | b1 b2 ... bn P P P ]

        into

                [ fn b1 ]

        while a naive approach would have selected

                [ P b1 ]

        as the final state.

        :param states: (seq_len, batch, num_directions * hidden_size)
        :param seq_lengths: integer list, whose element i represents the length of the input sequence
                            originally associated to the i-th state sequence in the 'states' minibatch.
        :return:  (batch, num_directions * hidden_size)
        """
        if seq_lengths is None or len(seq_lengths) == 1:
            return states[-1, :, :]

        max_seq_len = states.shape[0]
        batch_size = states.shape[1]

        if self.bidirectional:
            states = states.view(max_seq_len, batch_size, 2, -1)

            final_states = []
            for i in range(len(seq_lengths)):
                fw = states[seq_lengths[i] - 1, i, 0, :]
                bw = states[0, i, 1, :]
                final_states += [torch.cat([fw, bw], dim=0)]

            return torch.stack(final_states)

        else:
            final_states = []
            for i in range(len(seq_lengths)):
                fw = states[seq_lengths[i] - 1, i, :]
                final_states += [fw]

            return torch.stack(final_states)

    def state_size(self):
        return self.output_size * self.num_directions


class CustomGRUCell(torch.jit.ScriptModule):
    __constants__ = ['bias']

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 f=torch.tanh):
        super(CustomGRUCell, self).__init__()

        self.bias = bias
        self.f = f
        self.output_size = output_size

        # Reset gate
        self.W_in_r = torch.nn.Parameter(torch.Tensor(output_size, input_size + (1 if bias else 0)))
        self.W_hat_r = torch.nn.Parameter(torch.Tensor(output_size, output_size))

        # Update gate
        self.W_in_z = torch.nn.Parameter(torch.Tensor(output_size, input_size + (1 if bias else 0)))
        self.W_hat_z = torch.nn.Parameter(torch.Tensor(output_size, output_size))

        # Reservoir
        self.W_in = torch.nn.Parameter(torch.Tensor(output_size, input_size + (1 if bias else 0)))
        self.W_hat = torch.nn.Parameter(torch.Tensor(output_size, output_size))

        self.reset_parameters()

    @torch.jit.script_method
    def forward(self, input: torch.Tensor, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)

        output: (batch, hidden_size)
        """

        # Add the bias column
        if self.bias:
            input = torch.cat([ input, torch.ones((input.size(0), 1), device=self.W_in.device) ], dim=1)

        r = torch.sigmoid( self.W_in_r @ input.t() + self.W_hat_r @ hidden.t() ).t()
        z = torch.sigmoid( self.W_in_z @ input.t() + self.W_hat_z @ hidden.t() ).t()
        h_tilde = self.f( self.W_in @ input.t() + self.W_hat @ (r * hidden).t() ).t()
        h = z*hidden + (1 - z)*h_tilde

        return h, h

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.output_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)


class CustomGRU(CustomRNNBase):

    def __init__(self,
                 input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0, bidirectional=False,
                 f=torch.tanh):

        def cell_provider(input_size_, hidden_size_):
            return CustomGRUCell(input_size_, hidden_size_, bias=bias, f=f)

        super(CustomGRU, self).__init__(cell_provider, input_size, hidden_size, num_layers=num_layers, bias=bias,
                                        batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
