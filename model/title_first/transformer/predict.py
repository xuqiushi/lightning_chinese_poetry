import torch

from model.base.seq2seq_predictor import Seq2seqPredictor
from model.title_first.transformer.trainer import Trainer

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
        "五哀詩 故殿中侍御史滎陽鄭公",
        "釋重顯",
        "和提舉遊上方南禪十韻",
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
