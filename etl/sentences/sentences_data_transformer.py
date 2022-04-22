import re
from typing import List


from etl.base.base_seq2seq_data_transformer import BaseSeq2seqDataTransformer
from etl.entity.poetry import Poetry
from etl.entity.seq2seq.data_transformer_parameter import DataTransformerParameter
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY, CH_SEP


class SentencesDataTransformer(BaseSeq2seqDataTransformer):
    def _poetry_callback(
        self, poetry: Poetry, src_batch: List[str], trg_batch: List[str]
    ):
        if poetry.paragraphs and len(poetry.paragraphs) > 1:
            for split_index in range(1, len(poetry.paragraphs)):
                src = "".join(poetry.paragraphs[:split_index])
                trg = "".join(poetry.paragraphs[split_index:split_index + 1])
                src_batch.append(src)
                trg_batch.append(trg)


if __name__ == "__main__":
    raw_data_transformer = SentencesDataTransformer(
        DataTransformerParameter(src_directory=TANG_SONG_SHI_DIRECTORY)
    )
    test_df = raw_data_transformer.get_raw_df()
    test_vocab = raw_data_transformer.get_vocab()
    raw_data_transformer.get_train_test()
    test_train_df, test_val_df = raw_data_transformer.get_train_test()
    print(test_train_df, test_val_df)
