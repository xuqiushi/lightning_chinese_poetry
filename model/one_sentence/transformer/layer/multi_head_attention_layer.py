import torch
from torch import nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(
        self, hid_dim: int, n_heads: int, dropout_ratio: float, device: torch.device
    ):
        """
        :param hid_dim: embedding的feature数
        :param n_heads: 头数
        :param dropout_ratio: dropout比例
        :param device: 设备
        """
        super(MultiHeadAttentionLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.dropout_ratio = dropout_ratio
        self.device = device

        assert hid_dim % n_heads == 0, "hid_dim 必须能被n_heads整除"

        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)  # 点积之后进行缩放的比例

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        q = self.fc_q(query)
        k = self.fc_k(key)
        v = self.fc_v(value)

        # [batch_size, str_len, hid_dim]
        # -> [batch_size, str_len, n_heads, head_dim]
        # -> [batch_size, n_heads, str_len, head_dim]
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # [batch_size, n_heads, str_len, str_len]
        energy = torch.matmul(q, k.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), v)
        # x = [batch size, n heads, query len, head dim]
        x = x.permute(0, 2, 1, 3).contiguous()
        # x = [batch size, query len, hid dim]
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention
