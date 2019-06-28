import torch
import torch.nn as nn


class SingleTargetAttention(nn.Module):

    def __init__(self, input_size):
        super(SingleTargetAttention, self).__init__()
        self.attn = nn.Linear(input_size, 1)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, states):
        """

        :param states: tensor of shape (seq_len, batch, input_size)
        :return: a tensor of shape (batch, input_size)
        """
        alpha = self.attn(states)  # (seq_len, batch, 1)
        if self.mask is not None:
            alpha.data.masked_fill_(self.mask, -float('inf'))
        alpha = torch.softmax(alpha, dim=0)
        weighted_states = (alpha * states).sum(dim=0)  # (batch, input_size)
        return weighted_states, alpha

    def output_features(self):
        return self.attn.in_features


class LinSelfAttention(nn.Module):

    def __init__(self, input_size, hidden_size=-1, r=1):
        """
        Implementation of the self-attention mechanism described in:
            Zhouhan Lin, Minwei Feng, Cícero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen Zhou, Yoshua Bengio.
            A Structured Self-Attentive Sentence Embedding. ICLR 2017

        :param input_size:
        :param hidden_size: size of the hidden layer. If -1, set hidden_size = input_size.
                            If None, do not use a hidden layer (i.e. you already applied it outside of this module).
        :param r:
        """
        super(LinSelfAttention, self).__init__()

        if hidden_size == -1:
            hidden_size = input_size

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.r = r

        if hidden_size is None:
            self.Ws2 = torch.nn.Linear(input_size, r)
        else:
            self.Ws1 = torch.nn.Linear(input_size, hidden_size)
            self.Ws2 = torch.nn.Linear(hidden_size, r)

    def forward(self, states):
        """

        :param states: tensor of shape (seq_len, batch, input_size)
        :return: a tensor of shape (batch, self.r * input_size) representing the output,
                 and a tensor of shape (batch, seq_len, self.r) representing the attention weights.
        """
        states = states.transpose(0, 1)  # Transform to (batch, seq_len, input_size)

        # A: (batch, seq_len, self.r)
        if self.hidden_size is None:
            A = torch.softmax(self.Ws2(states), dim=1)
        else:
            A = torch.softmax(self.Ws2(torch.tanh(self.Ws1(states))), dim=1)

        # M: (batch, self.r, input_size)
        M = torch.bmm(A.transpose(1, 2), states)

        # Squash M into shape (batch, self.r * input_size)
        return M.view(states.size(0), -1), A

    def output_features(self):
        return self.input_size * self.r

    @staticmethod
    def loss_penalization(A, reg=1):
        """

        :param A: tensor of shape (batch, seq_len, self.r) representing the attention weights
        :param reg:
        :return:
        """
        AT = A.transpose(1, 2)
        identity = torch.eye(A.size(1), device=A.device)
        identity = identity.expand(A.size(0), A.size(1), A.size(1))
        penal = l2_matrix_norm(A@AT - identity)
        return reg * penal / A.size(0)


def l2_matrix_norm(m):
    """
    Frobenius norm calculation

    Args:
       m: {Variable}

    """
    return torch.sum(torch.sum(torch.sum(m ** 2, 1), 1) ** 0.5).type(torch.float)
