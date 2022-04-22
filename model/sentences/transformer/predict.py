import torch

from model.base.seq2seq_predictor import Seq2seqPredictor
from model.sentences.transformer.trainer import Trainer

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
        "廉可繼懸魚，官清術有餘。",
        "廉可繼懸魚，官清術有餘。民田侵廢苑，公署似閑居。",
        "廉可繼懸魚，官清術有餘。民田侵廢苑，公署似閑居。草長通囹圄，花飛落簿書。",
        # "見君松操直，經考只如初。"
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
