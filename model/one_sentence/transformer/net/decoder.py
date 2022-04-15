import torch
from torch import nn

from model.one_sentence.transformer.layer.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        hid_dim: int,
        n_layers: int,
        n_heads: int,
        pf_dim: int,
        dropout_ratio: float,
        device: torch.device,
        max_length: int = 100,
    ):
        """
        :param output_dim: 输出的字种类数
        :param hid_dim: 输出的embedding特征数
        :param n_layers: decoder的decoder层数，可以不同于encoder
        :param n_heads: 输出的注意力头数
        :param pf_dim: position wise ffn中间转换的特征数
        :param dropout_ratio: dropout 比例
        :param device: 设备
        :param max_length: 输出最长长度
        """
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_ratio = dropout_ratio
        self.device = device
        self.max_length = max_length

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList(
            [
                DecoderLayer(hid_dim, n_heads, pf_dim, dropout_ratio, device)
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """

        :param trg: 输出的字符串tokens [batch size, trg len]
        :param enc_src: encoder输出的结果 [batch size, src len, hid dim]
        :param trg_mask: 输出的mask [batch size, 1, trg len, trg len]
        :param src_mask: 输入的mask [batch size, 1, 1, src len]
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        tok_embedding = self.tok_embedding(trg) * self.scale
        pos_embedding = self.pos_embedding(pos)

        trg = self.dropout(tok_embedding + pos_embedding)

        attention = None
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        output = self.fc_out(trg)

        return output, attention
