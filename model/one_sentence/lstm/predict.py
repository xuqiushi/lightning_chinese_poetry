import torch

from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.vocab_loader import BOS, EOS, VocabLoader
from etl.one_sentence.custom_dataset import CustomDataset
from etl.one_sentence.one_sentence_loader import OneSentenceLoader
from model.one_sentence.lstm.net import Net


class Predict:
    @classmethod
    def predict_sentence(cls, sentence, t_transform, vocab, model, device, max_len=50):

        model.eval()

        src_indexes = t_transform(sentence)

        src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

        src_len = torch.LongTensor([len(src_indexes)])

        with torch.no_grad():
            hidden, cell = model.encoder(src_tensor)

        trg_indexes = [vocab[BOS]]

        for i in range(max_len):

            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

            with torch.no_grad():
                output, hidden, cell = model.decoder(trg_tensor, hidden, cell)

            pred_token = output.argmax(1).item()

            trg_indexes.append(pred_token)

            if pred_token == vocab[EOS]:
                break

        trg_tokens = [vocab.lookup_token(i) for i in trg_indexes]

        return trg_tokens[1:]


if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data_loader = OneSentenceLoader(TANG_SONG_SHI_DIRECTORY, device=test_device)
    vocab = VocabLoader(CustomDataset(TANG_SONG_SHI_DIRECTORY)).load_model()
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 10
    CLIP = 1
    test_model = Net(
        input_dim=INPUT_DIM,
        encode_emb_dim=ENC_EMB_DIM,
        encode_dropout=ENC_DROPOUT,
        output_dim=OUTPUT_DIM,
        decode_emb_dim=DEC_EMB_DIM,
        decode_dropout=DEC_DROPOUT,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        device=test_device,
    ).to(test_device)
    test_model.load_state_dict(
        torch.load(
            "/Users/xuqiushi/workspace/lightning_chinese_poetry/data/model/one_sentence/lstm/trainer/tut1-model.pt",
            map_location=torch.device("cpu"),
        )
    )
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
        "十金易一筆，"

    ]
    for test_string in test_string_list:
        predict_string = "".join(
            Predict.predict_sentence(
                test_string,
                test_data_loader.t_sequential,
                test_data_loader.vocab,
                test_model,
                test_device,
            )
        )
        print(
            f"{('手写：'+test_string):<20}".translate(full),
            f"{('机器接：'+predict_string):<50}".translate(full),
        )
