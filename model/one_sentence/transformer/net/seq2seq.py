import torch
from torch import nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src: torch.Tensor):
        """

        :param src: [batch size, src len]
        """
        # [batch size, 1, 1, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        return src_mask

    def make_trg_mask(self, trg: torch.Tensor):
        """

        :param trg: [batch size, trg len]
        :return:
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        # [trg len, trg len]生成一个左下角矩阵全是1的矩阵，这样来保证目标单词做自注意力的时候每个词都只能看到他前边的词。
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()

        # 广播机制先将trg_sub_mask扩展为[1, 1, trg len, trg len], 然后将1扩展。注意广播在增加维度的时候从后向前。
        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]
        return trg_mask

    def forward(self, src: torch.Tensor, trg: torch.Tensor):
        """

        :param src: [batch size, src len]
        :param trg: [batch size, trg len]
        """
        src_mask = self.make_src_mask(src)  # [batch size, 1, 1, src len]
        trg_mask = self.make_trg_mask(trg)  # [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # output = [batch size, trg len, output dim], attention = [batch size, n heads, trg len, src len]
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output, attention
