import torch
from common.base_models.esn import ESNBase, ESNMultiringCell
from common.base_models.attention import SingleTargetAttention, LinSelfAttention
from typing import List


class LeakyESNAttention(torch.nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 reservoir_size: int = 100,
                 num_esn_layers: int = 1,
                 mlp_n_hidden: int = 1,
                 mlp_hidden_size: int = 100,
                 dropout: float = 0,
                 attention_type: str = 'LinSelfAttention',
                 attention_hidden_size: int = 100,
                 attention_heads: int = 1,
                 scale_rec: List[float] = (1,),
                 scale_rec_bw: List[float] = (1,),
                 scale_in: List[float] = (1,),
                 scale_in_bw: List[float] = (1,),
                 density_in: List[float] = (1,),
                 density_in_bw: List[float] = (1,),
                 leaking_rate: List[float] = (1,),
                 leaking_rate_bw: List[float] = (1,)
                 ):
        super(LeakyESNAttention, self).__init__()

        bidirectional = True

        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = num_esn_layers
        self.reservoir_size = reservoir_size
        self.dropout = dropout
        self.attention_type = attention_type
        self.mlp_n_hidden = mlp_n_hidden
        self.mlp_hidden_size = mlp_hidden_size

        num_directions = 2 if bidirectional else 1

        def cell_provider(input_size_, reservoir_size_, layer, direction):
            return ESNMultiringCell(
                input_size_,
                reservoir_size_,
                bias=True,
                contractivity_coeff=scale_rec[layer] if direction == 0 else scale_rec_bw[layer],
                scale_in=scale_in[layer] if direction == 0 else scale_in_bw[layer],
                density_in=density_in[layer] if direction == 0 else density_in_bw[layer],
                leaking_rate=leaking_rate[layer] if direction == 0 else leaking_rate_bw[layer]
            )

        self.esn = ESNBase(
            cell_provider,
            input_size,
            reservoir_size,
            num_layers=self.n_layers,
            bidirectional=bidirectional
        )

        if self.attention_type == 'LinSelfAttention':
            self.ff1 = torch.nn.Linear(self.n_layers * num_directions * reservoir_size, attention_hidden_size)
            self.attn = LinSelfAttention(attention_hidden_size, r=attention_heads)
            mlp_input_size = self.attn.output_features()
        elif self.attention_type == 'Attention':
            self.ff1 = torch.nn.Linear(self.n_layers * num_directions * reservoir_size, attention_hidden_size)
            self.attn = SingleTargetAttention(attention_hidden_size)
            mlp_input_size = self.attn.output_features()
        elif self.attention_type == 'MaxPooling':
            self.ff1 = torch.nn.Linear(self.n_layers * num_directions * reservoir_size, attention_hidden_size)
            mlp_input_size = attention_hidden_size
            self.esn_bn = torch.nn.BatchNorm1d(mlp_input_size)
        elif self.attention_type == 'None':
            mlp_input_size = self.n_layers * num_directions * reservoir_size
        elif self.attention_type == 'Mean':
            mlp_input_size = self.n_layers * num_directions * reservoir_size
        else:
            raise Exception("Invalid attention type: " + self.attention_type)

        if mlp_n_hidden == 0:
            self.mlp_hn = torch.nn.ModuleList([])
            self.mlp_out = torch.nn.Linear(mlp_input_size, output_size)
        else:
            mlp_h1 = torch.nn.Linear(mlp_input_size, mlp_hidden_size)
            self.mlp_hn = torch.nn.ModuleList([torch.nn.Linear(mlp_hidden_size, mlp_hidden_size)
                                               for _ in range(mlp_n_hidden - 1)])
            self.mlp_hn.insert(0, mlp_h1)
            self.mlp_out = torch.nn.Linear(mlp_hidden_size, output_size)

        self._attnweights = None  # Attention weights for the latest minibatch.

    def readout(self, states: torch.Tensor):
        """
        :param states: (seq_len, batch_size, num_directions * hidden_size)
        :return:
        """
        s = states
        for lyr in self.mlp_hn:
            s = torch.nn.functional.dropout(s, p=self.dropout)
            s = torch.relu(lyr(s))

        s = self.mlp_out(s)
        return s

    def forward(self, input: torch.Tensor, seq_lengths=None):
        """
        input: (seq_len, batch_size, input_size)
        lengths: integer list of lengths, one for each sequence in 'input'. If provided, padding states
                 are automatically ignored.
        output: (1,)
        """
        seq_len = input.size(0)
        batch = input.size(1)

        if self.attention_type == 'LinSelfAttention' or self.attention_type == 'Attention':
            n_attn = self.ff1.out_features

            states, _ = self.esn.forward(input)  # states: (seq_len, batch, num_directions * hidden_size)
            states = torch.tanh(self.ff1(states))  # states: (seq_len, batch, n_attn)

            ## Let the recurrent network compute the states
            #states = torch.empty((seq_len, batch, n_attn), device=input.device)
            #for i, x in enumerate(self.esn.forward_long_sequence_yld(input)):
            #    # x: (num_layers * num_directions, batch, hidden_size)
            #    x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
            #    # Reduce dimensionality. x: (batch, n_attention)
            #    x = torch.tanh(self.ff1(x))
            #    states[i] = x

            # Apply Attention
            x, self._attnweights = self.attn.forward(states)  # x: (batch, n_attention)

        elif self.attention_type == 'MaxPooling':
            s = torch.zeros((batch, self.ff1.out_features), device=input.device)
            for i, x in enumerate(self.esn.forward_long_sequence_yld(input)):
                # x: (num_layers * num_directions, batch, hidden_size)
                x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
                # x: (batch, num_layers * num_directions * hidden_size)
                x = torch.tanh(self.ff1(x))
                s = torch.max(torch.stack((s, x)), 0)[0]
            x = self.esn_bn(s)

        elif self.attention_type == 'Mean':
            n_res = self.esn.num_layers * self.esn.num_directions * self.esn.reservoir_size

            # Let the recurrent network compute the states
            states = torch.empty((batch, n_res), device=input.device)
            count = 0
            for i, x in enumerate(self.esn.forward_long_sequence_yld(input)):
                # x: (num_layers * num_directions, batch, hidden_size)
                x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
                states += x
                count += 1
            x = states / count

        elif self.attention_type == 'None':
            x = self.esn.forward_long_sequence(input, seq_lengths=seq_lengths)
            x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)

        return self.readout(x)

    def loss_penalty(self):
        if self.attention_type == 'LinSelfAttention' and self.attn.r > 1:
            return LinSelfAttention.loss_penalization(self._attnweights)
        return 0
