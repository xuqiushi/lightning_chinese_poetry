import re
from typing import List


from etl.base.base_seq2seq_data_transformer import BaseSeq2seqDataTransformer
from etl.entity.poetry import Poetry
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY, CH_SEP


class RawDataTransformer(BaseSeq2seqDataTransformer):
    def _poetry_callback(
        self, poetry: Poetry, src_batch: List[str], trg_batch: List[str]
    ):
        for paragraph in poetry.paragraphs:
            sentences = re.findall(
                rf"[^{CH_SEP}]+[{CH_SEP}]",
                paragraph,
            )
            if len(sentences) < 2:
                continue
            else:
                for split_index in range(1, len(sentences)):
                    src = "".join(sentences[:split_index])
                    trg = "".join(sentences[split_index:])
                    src_batch.append(src)
                    trg_batch.append(trg)


if __name__ == "__main__":
    raw_data_transformer = RawDataTransformer(TANG_SONG_SHI_DIRECTORY)
    test_df = raw_data_transformer.get_raw_df()
    test_vocab = raw_data_transformer.get_vocab()
    raw_data_transformer.get_train_test()
    test_train_df, test_val_df = raw_data_transformer.get_train_test()
    print(test_train_df, test_val_df)
