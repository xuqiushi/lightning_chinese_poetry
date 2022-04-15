import pathlib

import torch

from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.vocab_loader import BOS, EOS, VocabLoader, PADDING
from etl.one_sentence.custom_iterable_dataset import CustomIterableDataset
from etl.one_sentence.one_sentence_loader import OneSentenceLoader
from model.one_sentence.lstm.net import Net
from model.one_sentence.transformer.net.decoder import Decoder
from model.one_sentence.transformer.net.encoder import Encoder
from model.one_sentence.transformer.net.seq2seq import Seq2Seq
from model.one_sentence.transformer.trainer import (
    BATCH_SIZE,
    STR_MAX_LENGTH,
    ENC_LAYERS,
    ENC_HEADS,
    ENC_PF_DIM,
    DEC_HEADS,
    DEC_LAYERS,
    DEC_PF_DIM,
    Trainer,
    HID_DIM,
    ENC_DROPOUT,
    DEC_DROPOUT,
)


class Predict:
    def __init__(self, data_directory: pathlib.Path):
        self.data_directory = data_directory

        self.vocab = VocabLoader(CustomIterableDataset(data_directory)).load_model()
        self.src_dim = len(self.vocab)
        self.trg_dim = len(self.vocab)
        self.src_pad_idx = self.vocab[PADDING]
        self.trg_pad_idx = self.vocab[PADDING]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = OneSentenceLoader(
            directory=self.data_directory,
            train_n_workers=8,
            train_batch_size=BATCH_SIZE,
            train_pre_fetch_factor=16,
            val_n_workers=8,
            val_batch_size=BATCH_SIZE,
            val_pre_fetch_factor=16,
            device=self.device,
            str_max_length=STR_MAX_LENGTH,
        )

    def predict_sentence(self, sentence, t_transform, vocab, device, max_len=100):
        enc = Encoder(
            self.src_dim,
            HID_DIM,
            ENC_LAYERS,
            ENC_HEADS,
            ENC_PF_DIM,
            ENC_DROPOUT,
            self.device,
            STR_MAX_LENGTH,
        )
        dec = Decoder(
            self.trg_dim,
            HID_DIM,
            DEC_LAYERS,
            DEC_HEADS,
            DEC_PF_DIM,
            DEC_DROPOUT,
            self.device,
            STR_MAX_LENGTH,
        )
        model = Seq2Seq(enc, dec, self.src_pad_idx, self.trg_pad_idx, self.device).to(
            self.device
        )
        model.load_state_dict(
            torch.load(
                Trainer(TANG_SONG_SHI_DIRECTORY).model_path,
                map_location=torch.device("cpu"),
            )
        )
        model.eval()
        src_indexes = t_transform(sentence)
        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
        src_mask = model.make_src_mask(src_tensor)

        with torch.no_grad():
            enc_src = model.encoder(src_tensor, src_mask)
        trg_indexes = [vocab[BOS]]

        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = model.decoder(
                    trg_tensor, enc_src, trg_mask, src_mask
                )
            pred_token = output.argmax(2)[:, -1].item()
            trg_indexes.append(pred_token)
            if pred_token == vocab[EOS]:
                break
        trg_tokens = [vocab.lookup_token(i) for i in trg_indexes]
        return trg_tokens[1:]


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_loader = OneSentenceLoader(
        directory=TANG_SONG_SHI_DIRECTORY,
        train_n_workers=8,
        train_batch_size=BATCH_SIZE,
        train_pre_fetch_factor=16,
        val_n_workers=8,
        val_batch_size=BATCH_SIZE,
        val_pre_fetch_factor=16,
        device=test_device,
        str_max_length=STR_MAX_LENGTH,
    )
    vocab = VocabLoader(CustomIterableDataset(TANG_SONG_SHI_DIRECTORY)).load_model()
    SPACE = "\N{IDEOGRAPHIC SPACE}"
    EXCLA = "\N{FULLWIDTH EXCLAMATION MARK}"
    TILDE = "\N{FULLWIDTH TILDE}"

    # strings of ASCII and full-width characters (same order)
    west = "".join(chr(i) for i in range(ord(" "), ord("~")))
    east = SPACE + "".join(chr(i) for i in range(ord(EXCLA), ord(TILDE)))

    # build the translation table
    full = str.maketrans(west, east)
    test_string_list = [
        "我自横刀向天笑，",
        "欲出未出光辣達，",
        "紅葉黄花三峽雨，",
        "林密山深客少過，",
        "日復一日去上班，",
        "花徑不曾緣客掃，",
        "花謝花飛花滿天，",
        "涼涼月色為你思念成河，",
        "風不起，",
        "得一夢來三事應，",
        "天生我材必有用，",
        "十金易一筆，",
    ]
    predictor = Predict(TANG_SONG_SHI_DIRECTORY)
    for test_string in test_string_list:
        predict_string = "".join(
            predictor.predict_sentence(
                test_string,
                test_data_loader.t_sequential,
                test_data_loader.vocab,
                test_device,
                100,
            )
        )
        print(
            f"{('手写：'+test_string):<20}".translate(full),
            f"{('机器接：'+predict_string):<50}".translate(full),
        )
