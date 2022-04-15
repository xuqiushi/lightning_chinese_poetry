import torch
from torch import nn

from model.one_sentence.transformer.layer.multi_head_attention_layer import MultiHeadAttentionLayer
from model.one_sentence.transformer.layer.position_wise_feedforward_layer import PositionWiseFeedforwardLayer


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hid_dim: int,
        n_heads: int,
        pf_dim: int,
        dropout_ratio: float,
        device: torch.device,
    ):
        super(EncoderLayer, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.dropout_ratio = dropout_ratio
        self.device = device

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout_ratio, device
        )
        self.position_wise_feedforward = PositionWiseFeedforwardLayer(
            hid_dim, pf_dim, dropout_ratio
        )
        self.dropout = nn.Dropout(dropout_ratio)

    def forward(self, src, src_mask):
        """
        :param src: [batch_size, src_len, hid_dim]
        :param src_mask: [batch_size, 1, 1, src_len], 第二维事实上是n_heads, 第三维是query_len, 第四维是key_len, 维度1会广播
        :return:
        """
        attention_result, _ = self.attention(src, src, src, src_mask)
        # attention -> dropout -> residual connection -> norm
        connected_attention_result = self.self_attn_layer_norm(
            src + self.dropout(attention_result)
        )
        position_wise_feedforward_result = self.position_wise_feedforward(
            connected_attention_result
        )
        # dropout -> residual connection -> norm
        connected_feedforward_result = self.ff_layer_norm(
            connected_attention_result + self.dropout(position_wise_feedforward_result)
        )

        return connected_feedforward_result
