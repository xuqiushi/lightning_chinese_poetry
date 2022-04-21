import torch

from model.base.seq2seq_predictor import Seq2seqPredictor
from model.one_sentence.transformer.trainer import Trainer

if __name__ == "__main__":
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    predictor = Seq2seqPredictor(Trainer())
    for test_string in test_string_list:
        predict_string = "".join(
            predictor.predict_sentence(
                test_string,
                test_device,
            )
        )
        print(
            f"{('手写：'+test_string):<20}".translate(full),
            f"{('机器接：'+predict_string):<50}".translate(full),
        )
