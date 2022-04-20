import json
import pathlib
import re
from typing import Tuple, List

import numpy as np
import torch
from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger
import pyarrow as pa
from pyarrow import Table
from sklearn.model_selection import train_test_split
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
        reset_train_val_df: bool = False,
        test_size: float = 0.2,
    ):
        self._src_directory = src_directory
        self._raw_tmp_data_path = self.class_tmp_directory() / "raw_tmp_data.arrow"
        self._vocab_path = self.class_data_directory() / "vocab.pt"
        self._train_df_path = self.class_data_directory() / "train.arrow"
        self._val_df_path = self.class_data_directory() / "val.arrow"
        self._reset_tmp_file = reset_tmp_file
        self._reset_vocab = reset_vocab
        self._reset_train_val_df = reset_train_val_df
        self._test_size = test_size

    @property
    def _need_reset_tmp(self):
        return not self._raw_tmp_data_path.exists() or self._reset_tmp_file

    @property
    def _need_reset_vocab(self):
        return not self._vocab_path.exists() or self._reset_vocab

    @property
    def _need_reset_train_val_df(self):
        return (
            not self._train_df_path.exists()
            or not self._val_df_path.exists()
            or self._reset_train_val_df
        )

    @classmethod
    def class_data_directory(cls):
        return DirectoryChanger.get_new_root_directory(
            pathlib.Path(__file__),
            config.directories.base_dir,
            config.directories.data_dir,
        )

    @classmethod
    def class_tmp_directory(cls):
        return DirectoryChanger.get_new_root_directory(
            pathlib.Path(__file__),
            config.directories.base_dir,
            config.directories.tmp_dir,
        )

    def get_raw_df(self) -> Table:
        """

        :return: schema [src: string trg: string]

        """
        if self._need_reset_tmp:
            self.save_raw_tmp_data()
        with pa.memory_map(str(self._raw_tmp_data_path), "rb") as source:
            return pa.ipc.open_file(source).read_all()

    def get_vocab(self) -> Vocab:
        if self._need_reset_vocab:
            self.save_vocab()
        return torch.load(self._vocab_path)

    def get_train_test(self) -> Tuple[Table, Table]:
        if self._need_reset_train_val_df:
            self.save_train_val_df()
        with pa.memory_map(str(self._train_df_path), "rb") as source:
            train_df = pa.ipc.open_file(source).read_all()

        with pa.memory_map(str(self._val_df_path), "rb") as source:
            val_df = pa.ipc.open_file(source).read_all()
        return train_df, val_df

    def save_raw_tmp_data(self) -> None:
        schema = pa.schema(
            [(self.COLUMN_NAME_SRC, pa.string()), (self.COLUMN_NAME_TRG, pa.string())]
        )
        with pa.OSFile(str(self._raw_tmp_data_path), "wb") as sink:
            with pa.ipc.new_file(sink, schema) as writer:
                file_path_list = list(self._src_directory.glob("*.json"))
                for file_path in tqdm(file_path_list):
                    src_batch = []
                    trg_batch = []
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
                                        src_batch.append(src)
                                        trg_batch.append(trg)
                    batch = pa.record_batch(
                        [
                            pa.array(src_batch, type=pa.string()),
                            pa.array(trg_batch, type=pa.string()),
                        ],
                        schema,
                    )
                    writer.write(batch)

    def save_vocab(self):
        def _vocab_iter():
            raw_df = self.get_raw_df()
            for record_index in range(len(raw_df)):
                yield list(raw_df[self.COLUMN_NAME_SRC][record_index].as_py())
                yield list(raw_df[self.COLUMN_NAME_TRG][record_index].as_py())

        vocab = build_vocab_from_iterator(
            tqdm(_vocab_iter()),
            specials=[UNKNOWN, BOS, PADDING, EOS],
            special_first=True,
        )
        vocab.set_default_index(vocab[UNKNOWN])
        torch.save(vocab, self._vocab_path)

    @classmethod
    def _save_sub_df(
        cls,
        df_raw: Table,
        select_index: List[int],
        save_path: pathlib.Path,
        vocab: Vocab,
    ):
        mask = np.zeros(len(df_raw), dtype=bool)
        mask[select_index] = True
        table: Table = df_raw.filter(pa.array(mask))
        schema = pa.schema(
            [
                (cls.COLUMN_NAME_SRC, pa.list_(pa.int32())),
                (cls.COLUMN_NAME_TRG, pa.list_(pa.int32())),
            ]
        )
        with pa.OSFile(str(save_path), "wb") as sink:
            with pa.ipc.new_file(
                sink,
                schema,
            ) as writer:
                src_batch = []
                trg_batch = []
                for batch in tqdm(table.to_batches()):
                    for sub_index in range(len(batch)):
                        src_batch.append(
                            vocab([BOS])
                            + vocab(list(batch[cls.COLUMN_NAME_SRC][sub_index].as_py()))
                            + vocab([EOS])
                        )
                        trg_batch.append(
                            vocab([BOS])
                            + vocab(list(batch[cls.COLUMN_NAME_TRG][sub_index].as_py()))
                            + vocab([EOS])
                        )
                batch = pa.record_batch(
                    [
                        pa.array(src_batch, type=pa.list_(pa.int32())),
                        pa.array(trg_batch, type=pa.list_(pa.int32())),
                    ],
                    schema,
                )
                writer.write(batch)

    def save_train_val_df(self):
        raw_df = self.get_raw_df()
        vocab = self.get_vocab()
        train_index, val_index, _, _ = train_test_split(
            range(len(raw_df)), range(len(raw_df)), test_size=self._test_size
        )
        self._save_sub_df(raw_df, train_index, self._train_df_path, vocab)
        self._save_sub_df(raw_df, val_index, self._val_df_path, vocab)


if __name__ == "__main__":
    raw_data_transformer = RawDataTransformer(TANG_SONG_SHI_DIRECTORY)
    test_df = raw_data_transformer.get_raw_df()
    test_vocab = raw_data_transformer.get_vocab()
    raw_data_transformer.get_train_test()
    test_train_df, test_val_df = raw_data_transformer.get_train_test()
    print(test_train_df, test_val_df)
