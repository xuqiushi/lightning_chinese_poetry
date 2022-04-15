import torch
from torch import nn


class PositionWiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim: int, pf_dim: int, dropout_ratio: float):
        super(PositionWiseFeedforwardLayer, self).__init__()
        self.pf_dim = pf_dim
        self.dropout_ratio = dropout_ratio

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        """
        :param x: [batch size, seq len, hid dim]
        """
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x
