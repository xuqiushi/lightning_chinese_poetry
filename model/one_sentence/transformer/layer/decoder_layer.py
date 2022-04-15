import torch
from torch import nn

from model.one_sentence.transformer.layer.multi_head_attention_layer import (
    MultiHeadAttentionLayer,
)
from model.one_sentence.transformer.layer.position_wise_feedforward_layer import (
    PositionWiseFeedforwardLayer,
)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout_ratio: float,
        device: torch.device,
    ):
        super(DecoderLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_ratio = dropout_ratio
        self.device = device

        self.self_attn_layer_norm = nn.LayerNorm(
            hid_dim
        )  # 我们只需要norm最后一个维度，所以只传最后一个维度的长度即可
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout_ratio, device
        )
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout_ratio, device
        )
        self.position_wise_feedforward = PositionWiseFeedforwardLayer(
            hid_dim, pf_dim, dropout_ratio
        )
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        """

        :param trg: [batch size, trg len, hid dim]
        :param enc_src: [batch size, src len, hid dim]
        :param trg_mask: [batch size, 1, trg len, trg len] 目标mask需要保证每个单词只有此单词前面的单词有权重
        :param src_mask: [batch size, 1, 1, src len]
        """
        trg_attention_result, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg_connected_attention_result = self.self_attn_layer_norm(
            trg + self.dropout(trg_attention_result)
        )
        src_attention_result, attention = self.encoder_attention(
            trg_connected_attention_result, enc_src, enc_src, src_mask
        )
        trg_src_connected_result = self.enc_attn_layer_norm(
            trg_connected_attention_result + self.dropout(src_attention_result)
        )
        position_wise_ffc = self.position_wise_feedforward(trg_src_connected_result)
        ffc_norm = self.ff_layer_norm(
            trg_src_connected_result + self.dropout(position_wise_ffc)
        )

        return ffc_norm, attention
