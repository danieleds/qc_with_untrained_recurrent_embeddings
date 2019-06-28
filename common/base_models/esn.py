import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Sequence


def to_sparse(tensor, density, sparse_repr=False) -> torch.Tensor:
    if density < 1:
        if sparse_repr:
            indices = (torch.rand_like(tensor) <= density).nonzero()
            values = torch.Tensor([ tensor[tuple(i)] for i in indices ])
            return torch.sparse.FloatTensor(indices.t(), values, tensor.size())
        else:
            return tensor * (torch.rand_like(tensor) <= density).type(tensor.dtype)
    else:
        return tensor


def random_matrix(size) -> torch.Tensor:
    return torch.rand(size)


class ESNMultiringCell(torch.jit.ScriptModule):
    __constants__ = ['scale', 'leaking_rate', 'bias']

    def __init__(self,
                 input_size,
                 output_size,
                 bias=True,
                 contractivity_coeff=0.9,
                 scale_in=1.0,
                 density_in=1.0,
                 f=torch.tanh,
                 leaking_rate=1.0,
                 input_w_distribution='uniform',  # Either "uniform" or "bernoulli"
                 rescaling_method='specrad',  # Either "norm" or "spectrad"
                 topology='multiring'):  # Either "ring" or "multiring"
        super(ESNMultiringCell, self).__init__()

        self.bias = bias
        self.contractivity_coeff = contractivity_coeff
        self.scale_in = scale_in
        self.f = f
        self.rescaling_method = rescaling_method
        self.topology = topology  # 'ring' | 'multiring'
        self.leaking_rate = leaking_rate

        # Reservoir
        self.W_in = random_matrix((output_size, input_size + (1 if bias else 0))) * 2 - 1
        self.W_in = to_sparse(self.W_in, density_in)
        if input_w_distribution == 'bernoulli':
            self.W_in[self.W_in >= 0] = +1
            self.W_in[self.W_in <  0] = -1

        # Sparsity
        if self.topology == 'multiring':
            self.ring = nn.Parameter(torch.randperm(output_size), requires_grad=False)
        elif self.topology == 'ring':
            self.ring = nn.Parameter(torch.cat([torch.arange(1, output_size), torch.tensor([0])]), requires_grad=False)
        else:
            raise Exception("Unimplemented topology: " + str(self.topology))

        # Scale W_in
        self.W_in = scale_in * self.W_in

        # Scale W_hat
        if rescaling_method == 'specrad':
            # No need to compute the spectral radius for ring or multiring topologies since
            # the spectral radius is equal to the value of the nonzero elements
            self.scale = contractivity_coeff
        else:
            raise Exception("Rescaling for ESNMultiringCell MUST be specrad!")

        # Assign as Parameters
        self.W_in = nn.Parameter(self.W_in, requires_grad=False)

    @torch.jit.script_method
    def forward(self, input, hidden):
        """
        input: (batch, input_size)
        hidden: (batch, hidden_size)

        output: (batch, hidden_size)
        """
        # Add the bias column
        if self.bias:
            input = torch.cat([ input, torch.ones((input.size(0), 1), device=self.W_in.device) ], dim=1)

        h_tilde = self.f( self.W_in @ input.t() + self.scale * hidden.t()[self.ring] ).t()
        h = (1-self.leaking_rate) * hidden + self.leaking_rate * h_tilde

        return h


class ESNBase(torch.nn.Module):

    def __init__(self, cell_provider: Callable[[int, int, int, int], torch.nn.Module],
                 input_size, reservoir_size, num_layers=1, dropout=0, bidirectional=False):
        super(ESNBase, self).__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # If True, the last state is automatically preserved for subsequent forward() calls
        self.preserve_last_state = False
        self.last_state = None

        if self.bidirectional and self.preserve_last_state:
            raise Exception("bidirectional=True and preserve_last_states=True are incompatible.")

        num_directions = 2 if bidirectional else 1

        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = input_size if layer == 0 else reservoir_size * num_directions
                suffix = '_reverse' if direction == 1 else ''

                # Instantiate a cell and save it as an instance on this RNN
                cell = cell_provider(layer_input_size, reservoir_size, layer, direction)
                setattr(self, 'cell_l{}{}'.format(layer, suffix), cell)

    def forward_long_sequence_yld(self, input, h_0=None):
        """
        For when you need to feed very long sequences and need all states, but one at a time.
        :param input: (seq_len, batch, input_size)
        :param h_0: (num_layers * num_directions, batch, hidden_size)

        Yields tensors of shape (num_directions, batch, hidden_size)
        """
        if self.num_layers > 1:
            raise Exception("forward_long_sequence is not compatible with multilayer reservoirs yet.")

        device = self._curr_device()

        batch = input.size(1)

        if h_0 is None:
            h_0 = torch.zeros((self.num_directions, batch, self.reservoir_size), device=device)

        h_l = [h_0[i] for i in range(self.num_directions)]

        cell_fw = self.get_cell(0, 0)
        cell_bw = self.get_cell(0, 1) if self.bidirectional else None
        for i in range(len(input)):
            step_fw = input[i]
            h_l[0] = cell_fw.forward(step_fw, h_l[0])  # (batch, hidden_size)

            if self.bidirectional:
                step_bw = input[-i-1]
                h_l[1] = cell_bw.forward(step_bw, h_l[1])  # (batch, hidden_size)

            if self.bidirectional:
                yield torch.stack((h_l[0], h_l[1]))
            else:
                yield h_l[0].unsqueeze(0)

    def forward_long_sequence(self, input, h_0=None, seq_lengths=None):
        """
        For when you need to feed very long sequences and you only care about the last states.
        :param input: (seq_len, batch, input_size)
        :param h_0: (num_layers * num_directions, batch, hidden_size)
        :return: h_n, which is a tensor of shape (num_directions, batch, hidden_size). It is compatible with the
                 output tensor h_n from the forward method, which has shape (num_layers * num_directions, batch, hidden_size).
                 It contains the hidden state for the last step of the sequence.
        """
        batch = input.size(1)

        last_forward_states = torch.zeros(batch, self.reservoir_size, device=input.device)
        last_backward_states = None  # (batch, self.reservoir_size)

        for i, h in enumerate(self.forward_long_sequence_yld(input, h_0=h_0)):
            if seq_lengths is not None:
                last_forward_states = h[0]
            else:
                # Find all batch indices which have been completely processed
                done_idx = [ j for j in range(batch) if seq_lengths[j] == i+1 ]
                last_forward_states[done_idx] = h[0, done_idx]
            if self.bidirectional:
                last_backward_states = h[1]

        if self.bidirectional:
            return torch.stack((last_forward_states, last_backward_states))
        else:
            return last_forward_states.unsqueeze(0)

    def forward(self, input, h_0=None):
        """

        :param input: tensor of shape (seq_len, batch, input_size)
        :param h_0: tensor of shape (num_layers * num_directions, batch, hidden_size)
        :param seq_lengths: length of each non-padded sequence in input.
        :param return_all_states: whether you need the state associated to each step of the sequence, or just the
                last one (i.e. for GPU memory limitations).
        :return: a tuple (output, h_n):
                - output: a tensor of shape (seq_len, batch, num_directions * hidden_size).
                    It contains only the output of the last layer, for both directions concatenated.
                    If return_all_states == False, it is of shape (1, batch, num_directions * hidden_size)
                    and only contains the last step of the sequence
                - h_n: a tensor of shape (num_layers * num_directions, batch, hidden_size).
                    It contains the hidden state for the last step, in all layers
        """
        if self.preserve_last_state and h_0 is None and self.last_state is not None:
            h_0 = self.last_state

        x, h = self._forward_type_2(input, h_0)

        if self.preserve_last_state:
            self.last_state = h

        return x, h

    def _forward_type_1(self, input, h_0=None):
        """
        input: (seq_len, batch, input_size)
        h_0: (num_layers * num_directions, batch, hidden_size)

        output: (seq_len, batch, num_directions * hidden_size)  # Only the output of the last layer, for both directions
                                                                  concatenated
        h_n: (num_layers * num_directions, batch, hidden_size)  # Hidden state for the last step, in all layers
        """
        # TODO: Deep Bidirectional implemented as two completely separate dynamics that are joined at the end
        # https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298
        raise Exception("Not implemented")

    def _forward_type_2(self, input, h_0=None):
        """
        input: (seq_len, batch, input_size)
        h_0: (num_layers * num_directions, batch, hidden_size)

        output: (seq_len, batch, num_directions * hidden_size)  # Only the output of the last layer, for both directions
                                                                  concatenated
        h_n: (num_layers * num_directions, batch, hidden_size)  # Hidden state for the last step, in all layers
        """
        # https://github.com/pytorch/pytorch/issues/4930#issuecomment-361851298

        device = self._curr_device()

        seq_len = input.size(0)
        batch = input.size(1)

        if h_0 is None:
            h_0 = torch.zeros((self.num_layers * self.num_directions, batch, self.reservoir_size), device=device)

        h_l = [h_0[i] for i in range(self.num_layers * self.num_directions)]

        # The reservoir activation for the whole sequence at all layers, for all time steps.
        prev_output = torch.zeros((self.num_layers, seq_len, batch, self.num_directions * self.reservoir_size),
                                  device=device)
        for l in range(self.num_layers):
            # shape: (seq_len, batch, num_directions * hidden_size)
            layer_input = input if l == 0 else prev_output[l-1]

            if l > 0 and self.dropout > 0:
                layer_input = F.dropout(layer_input, p=self.dropout)

            forward_states = []  # list of seq_len tensors of size (batch, hidden_size)
            backward_states = []  # list of seq_len tensors of size (batch, hidden_size)

            # Forward pass
            cell = getattr(self, 'cell_l{}'.format(l))
            for step in layer_input:
                input_state = h_l[self.num_directions * l + 0]
                h = cell.forward(step, input_state)  # (batch, hidden_size)
                h_l[self.num_directions * l + 0] = h
                forward_states = forward_states + [h]

            if self.bidirectional:
                # Backward pass
                cell = getattr(self, 'cell_l{}_reverse'.format(l))
                for step in reversed(layer_input):
                    input_state = h_l[self.num_directions * l + 1]
                    h = cell.forward(step, input_state)  # (batch, hidden_size)
                    h_l[self.num_directions * l + 1] = h
                    backward_states = [h] + backward_states

            if self.bidirectional:
                prev_output_forward = torch.stack(forward_states)  # (seq_len, batch, hidden_size)
                prev_output_backward = torch.stack(backward_states)  # (seq_len, batch, hidden_size)
                prev_output[l] = torch.cat((prev_output_forward, prev_output_backward), dim=2)
            else:
                prev_output[l] = torch.stack(forward_states)

        return prev_output[-1], torch.stack(h_l)

    def extract_last_states(self, states, seq_lengths=None):
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

    # def state_size(self):
    #     return self.reservoir_size * self.num_directions

    def _curr_device(self):
        return next(self.parameters()).device

    def get_cell(self, layer: int, direction: int = 0) -> torch.nn.Module:
        """
        Returns one of the cells in this model
        :param layer: starts from zero
        :param direction:
        :return:
        """
        suffix = '_reverse' if direction == 1 else ''
        return getattr(self, 'cell_l{}{}'.format(layer, suffix))

    def get_cells(self) -> Sequence[torch.nn.Module]:
        """
        Returns all cells in this model
        :return:
        """
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                yield self.get_cell(layer, direction=direction)

