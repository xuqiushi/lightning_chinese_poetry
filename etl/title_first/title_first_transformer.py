import re
from typing import List


from etl.base.base_seq2seq_data_transformer import BaseSeq2seqDataTransformer
from etl.entity.poetry import Poetry
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY, CH_SEP


class TitleFirstTransformer(BaseSeq2seqDataTransformer):
    def _poetry_callback(
        self, poetry: Poetry, src_batch: List[str], trg_batch: List[str]
    ):
        if poetry.paragraphs:
            first_paragraph = poetry.paragraphs[0]
            first_half_sentences = re.findall(
                rf"[^{CH_SEP}]+[{CH_SEP}]",
                first_paragraph,
            )
            if poetry.title and first_half_sentences:
                src_batch.append(poetry.title)
                trg_batch.append(first_half_sentences[0])


if __name__ == "__main__":
    raw_data_transformer = TitleFirstTransformer(TANG_SONG_SHI_DIRECTORY)
    test_df = raw_data_transformer.get_raw_df()
    test_vocab = raw_data_transformer.get_vocab()
    raw_data_transformer.get_train_test()
    test_train_df, test_val_df = raw_data_transformer.get_train_test()
    print(test_train_df, test_val_df)
