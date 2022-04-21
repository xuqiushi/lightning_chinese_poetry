import torch

from etl.etl_contants import PADDING, BOS, EOS
from model.base.seq2seq_trainer import Seq2seqTrainer


class Seq2seqPredictor:
    def __init__(self, seq2seq_trainer: Seq2seqTrainer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = seq2seq_trainer.data_loader
        self.model = seq2seq_trainer.load_model(self.device)
        self.model.eval()
        self.str_max_length = seq2seq_trainer.transformer_model_parameter.str_max_length
        self.vocab = self.data_loader.vocab
        self.src_dim = len(self.vocab)
        self.trg_dim = len(self.vocab)
        self.src_pad_idx = self.vocab[PADDING]
        self.trg_pad_idx = self.vocab[PADDING]

    def predict_sentence(self, sentence, device):
        src_indexes = self.vocab([BOS]) + self.vocab(list(sentence)) + self.vocab([EOS])
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = self.model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = self.model.encoder(src_tensor, src_mask)
        trg_indexes = [self.vocab[BOS]]

        for i in range(self.str_max_length):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = self.model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = self.model.decoder(
                    trg_tensor, enc_src, trg_mask, src_mask
                )
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            if pred_token == self.vocab[EOS]:
                break
        trg_tokens = [self.vocab.lookup_token(i) for i in trg_indexes]
        return trg_tokens[1:]
