import torch

from etl.base.base_seq2seq_data_transformer import BaseSeq2seqDataTransformer
from etl.etl_contants import POETRY_END
from model.base.seq2seq_predictor import Seq2seqPredictor
from model.one_sentence.transformer.trainer import Trainer as OneSentenceTrainer
from model.title_first.transformer.trainer import Trainer as TitleFirstTrainer
from model.sentences.transformer.trainer import Trainer as SentencesTrainer


class Writer:
    def write(self, title: str):
        pass


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
    # title_first_trainer = TitleFirstTrainer()
    # one_sentence_trainer = OneSentenceTrainer()
    # sentences_trainer = SentencesTrainer()
    title_first_predictor = Seq2seqPredictor(TitleFirstTrainer())
    one_sentence_predictor = Seq2seqPredictor(OneSentenceTrainer())
    sentences_predictor = Seq2seqPredictor(SentencesTrainer())
    titles = [
        "上班",
        "孟建辰",
        "徐秋實",
        "崔文鵬",
        "于濤",
        "谷澤霖",
        "喬梁",
        "劉睿"
    ]
    for title in titles:
        first_half = "".join(
            title_first_predictor.predict_sentence(title, test_device)[:-1]
        )
        first_sentence_last_half = "".join(
            one_sentence_predictor.predict_sentence(first_half, test_device)[:-1]
        )
        sentences = [first_half + first_sentence_last_half]
        count = 0
        should_stop = False
        while not should_stop:
            if count >= 4:
                should_stop = True
                sentences.append("未完待续")
                continue
            next_sentence_list = sentences_predictor.predict_sentence(
                "".join(sentences), test_device
            )
            should_stop = next_sentence_list[-2] == POETRY_END
            next_first_half = ""
            for chara in next_sentence_list:
                if chara != "，":
                    next_first_half += chara
                else:
                    next_first_half += "，"
                    break
            next_second_half = "".join(one_sentence_predictor.predict_sentence(next_first_half, test_device)[:-1])
            # if should_stop:
            #     next_sentence = "".join(next_sentence_list[:-2])
            # else:
            #     next_sentence = "".join(next_sentence_list[:-1])

            sentences.append(next_first_half + next_second_half)
            count += 1

        print(f"{'='.translate(full)*50}")
        print(
            f"{title:^50}".translate(full),
        )
        for sentence in sentences:
            print(f"{sentence:^50}".translate(full))
