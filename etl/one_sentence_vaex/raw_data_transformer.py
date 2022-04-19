import csv
import json
import pathlib
import re

import torch
import vaex
from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm

from etl.entity.poetry import Poetry
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY, CH_SEP, UNKNOWN, BOS, PADDING, EOS
from config import config


class RawDataTransformer:
    COLUMN_NAME_SRC = "src"
    COLUMN_NAME_TRG = "trg"

    def __init__(
        self,
        src_directory: pathlib.Path,
        reset_tmp_file: bool = False,
        reset_vocab: bool = False,
    ):
        self._src_directory = src_directory
        self._raw_tmp_data_path = self.class_tmp_directory() / "raw_tmp_data.csv"
        self._vocab_path = self.class_data_directory() / "vocab.pt"
        self._reset_tmp_file = reset_tmp_file
        self._reset_vocab = reset_vocab

    @property
    def _need_reset_tmp(self):
        return not self._raw_tmp_data_path.exists() or self._reset_tmp_file

    @property
    def _need_reset_vocab(self):
        return not self._vocab_path.exists() or self._reset_vocab

    @classmethod
    def class_data_directory(cls):
        return DirectoryChanger.get_new_root_directory(
            pathlib.Path(__file__),
            config.directories.base_dir,
            config.directories.tmp_dir,
        )

    @classmethod
    def class_tmp_directory(cls):
        return DirectoryChanger.get_new_root_directory(
            pathlib.Path(__file__),
            config.directories.base_dir,
            config.directories.tmp_dir,
        )

    def get_raw_df(self) -> vaex.dataframe.DataFrameLocal:
        if self._need_reset_tmp:
            self.save_raw_tmp_data()
        return vaex.open(str(self._raw_tmp_data_path))

    def get_vocab(self) -> Vocab:
        if self._need_reset_vocab:
            self.save_vocab()
        return torch.load(self._vocab_path)

    def save_vocab(self):
        def _vocab_iter():
            for start_index, end_index, chunk in self.get_raw_df().to_records(
                array_type="list", chunk_size=1000
            ):
                for record in chunk:
                    yield record[self.COLUMN_NAME_SRC]
                    yield record[self.COLUMN_NAME_TRG]

        vocab = build_vocab_from_iterator(
            tqdm(_vocab_iter()),
            specials=[UNKNOWN, BOS, PADDING, EOS],
            special_first=True,
        )
        vocab.set_default_index(vocab[UNKNOWN])
        torch.save(vocab, self._vocab_path)

    def save_raw_tmp_data(self) -> vaex.dataframe:
        file_path_list = list(self._src_directory.glob("*.json"))
        with open(self._raw_tmp_data_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow([self.COLUMN_NAME_SRC, self.COLUMN_NAME_TRG])
            for file_path in tqdm(file_path_list):
                with open(file_path, "r") as f:
                    current_json = json.load(f)
                    for item_index, item in enumerate(current_json):
                        poetry = Poetry.parse_obj(item)
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
                                    writer.writerow([list(src), list(trg)])


if __name__ == "__main__":
    raw_data_transformer = RawDataTransformer(TANG_SONG_SHI_DIRECTORY)
    test_df = raw_data_transformer.get_raw_df()
    test_vocab = raw_data_transformer.get_vocab()
    print(test_vocab)
