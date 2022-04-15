import random

import torch.nn
from torch import nn


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        encode_emb_dim,
        encode_dropout,
        output_dim,
        decode_emb_dim,
        decode_dropout,
        hid_dim,
        n_layers,
        device
    ):
        super(Net, self).__init__()
        self.encoder = Net.Encoder(
            input_dim, encode_emb_dim, hid_dim, n_layers, encode_dropout
        )
        self.decoder = Net.Decoder(
            output_dim, decode_emb_dim, hid_dim, n_layers, decode_dropout
        )
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        word_of_target = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(word_of_target, hidden, cell)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            word_of_target = trg[:, t] if teacher_force else top1
        return outputs

    class Encoder(nn.Module):
        def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
            super().__init__()
            self.input_dim = input_dim
            self.emb_dim = emb_dim
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            self.dropout = nn.Dropout(dropout)
            self.embedding = torch.nn.Embedding(input_dim, emb_dim)
            self.rnn = nn.LSTM(
                emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
            )

        def forward(self, src):
            """
            :param src: [N, L, H_{in}]
            :return:
            """
            embedded = self.dropout(self.embedding(src))
            outputs, (hidden, cell) = self.rnn(embedded)
            return hidden, cell

    class Decoder(nn.Module):
        def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
            super().__init__()
            self.output_dim = output_dim
            self.hid_dim = hid_dim
            self.n_layers = n_layers
            self.embedding = nn.Embedding(output_dim, emb_dim)
            self.rnn = nn.LSTM(
                emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True
            )
            self.fc_out = nn.Linear(hid_dim, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, one_of_target: torch.Tensor, hidden, cell):
            one_of_target = one_of_target.unsqueeze(1)
            embedded = self.dropout(self.embedding(one_of_target))
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            prediction = self.fc_out(output.squeeze(1))
            return prediction, hidden, cell
