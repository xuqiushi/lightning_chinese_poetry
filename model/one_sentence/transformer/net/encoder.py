import torch
from torch import nn

from model.one_sentence.transformer.layer.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout_radio: float,
        device: torch.device,
        max_length: int = 100,
    ):
        """
        :param input_dim: 输入特征维度，即token种类数
        :param hid_dim: tok_embedding与pos_embedding的特征维度
        :param n_layers: encoder layer 数量
        :param n_heads: 注意力头数
        :param pf_dim: position wise feedforward层的中间变换特征维度
        :param dropout_radio: dropout概率
        :param device: 设备
        :param max_length: 最大长度
        """
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_radio = dropout_radio
        self.device = device
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(hid_dim, n_heads, pf_dim, dropout_radio, device)
                for _ in range(n_layers)
            ]
        )
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim], device=device))  # pos_embedding需要乘以这个，据说可以减少方差

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        """
        :param src: [batch_size src_len]
        :param src_mask: [batch_size, 1, 1, src_len], 第二维事实上是n_heads, 第三维是query_len, 第四维是key_len, 维度1会广播
        :return:
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        # range(0, src_len) [src_len] -> [1, src_len] -> [batch_size, src_len]
        pos = (
            torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        )

        tok_embedding = self.tok_embedding(src) * self.scale
        pos_embedding = self.pos_embedding(pos)

        embedded_input = tok_embedding + pos_embedding
        encoded_result = embedded_input  # [batch_size, src_len, hid_dim]

        for layer in self.layers:
            encoded_result = layer(encoded_result, src_mask)

        return encoded_result
