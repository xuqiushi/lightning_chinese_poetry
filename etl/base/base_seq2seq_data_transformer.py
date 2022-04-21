from abc import ABCMeta, abstractmethod
import json
import pathlib
from typing import Tuple, List

import numpy as np
import torch
from lightning_fast.tools.path_tools.class_directory import ClassDirectory
from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger
import pyarrow as pa
from pyarrow import Table
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm

from etl.entity.poetry import Poetry
from etl.etl_contants import UNKNOWN, BOS, PADDING, EOS
from config import config


class BaseSeq2seqDataTransformer(metaclass=ABCMeta):
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
        self._raw_tmp_data_path = self.class_tmp_directory / "raw_tmp_data.arrow"
        self._vocab_path = self.class_data_directory / "vocab.pt"
        self._train_df_path = self.class_data_directory / "train.arrow"
        self._val_df_path = self.class_data_directory / "val.arrow"
        self._reset_tmp_file = reset_tmp_file
        self._reset_vocab = reset_vocab
        self._reset_train_val_df = reset_train_val_df
        self._test_size = test_size

    def get_raw_df(self) -> Table:
        """

        :return: schema [src: string trg: string]

        """
        if self._need_reset_tmp:
            self._save_raw_tmp_data()
        with pa.memory_map(str(self._raw_tmp_data_path), "rb") as source:
            return pa.ipc.open_file(source).read_all()

    def get_vocab(self) -> Vocab:
        if self._need_reset_vocab:
            self._save_vocab()
        return torch.load(self._vocab_path)

    def get_train_test(self) -> Tuple[Table, Table]:
        if self._need_reset_train_val_df:
            self._save_train_val_df()
        with pa.memory_map(str(self._train_df_path), "rb") as source:
            train_df = pa.ipc.open_file(source).read_all()

        with pa.memory_map(str(self._val_df_path), "rb") as source:
            val_df = pa.ipc.open_file(source).read_all()
        return train_df, val_df

    @abstractmethod
    def _poetry_callback(
        self, poetry: Poetry, src_batch: List[str], trg_batch: List[str]
    ):
        pass

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

    @property
    def class_data_directory(self):
        return DirectoryChanger.get_new_root_directory(
            ClassDirectory(self.__class__).get_class_path(),
            config.directories.base_dir,
            config.directories.data_dir,
        )

    @property
    def class_tmp_directory(self):
        return DirectoryChanger.get_new_root_directory(
            ClassDirectory(self.__class__).get_class_path(),
            config.directories.base_dir,
            config.directories.tmp_dir,
        )

    def _save_raw_tmp_data(self) -> None:
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
                            self._poetry_callback(
                                poetry=poetry, src_batch=src_batch, trg_batch=trg_batch
                            )
                    batch = pa.record_batch(
                        [
                            pa.array(src_batch, type=pa.string()),
                            pa.array(trg_batch, type=pa.string()),
                        ],
                        schema,
                    )
                    writer.write(batch)

    def _save_vocab(self):
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
        schema = pa.schema(
            [
                (cls.COLUMN_NAME_SRC, pa.list_(pa.int64())),
                (cls.COLUMN_NAME_TRG, pa.list_(pa.int64())),
            ]
        )
        with pa.OSFile(str(save_path), "wb") as sink:
            with pa.ipc.new_file(
                sink,
                schema,
            ) as writer:
                for index in select_index:
                    src_batch = []
                    trg_batch = []
                    src_batch.append(
                        vocab([BOS])
                        + vocab(list(df_raw[cls.COLUMN_NAME_SRC][index].as_py()))
                        + vocab([EOS])
                    )
                    trg_batch.append(
                        vocab([BOS])
                        + vocab(list(df_raw[cls.COLUMN_NAME_TRG][index].as_py()))
                        + vocab([EOS])
                    )
                    batch = pa.record_batch(
                        [
                            pa.array(src_batch, type=pa.list_(pa.int64())),
                            pa.array(trg_batch, type=pa.list_(pa.int64())),
                        ],
                        schema,
                    )
                    writer.write(batch)

    def _save_train_val_df(self):
        raw_df = self.get_raw_df()
        vocab = self.get_vocab()
        shuffled_index = shuffle(range(len(raw_df)))
        train_index, val_index, _, _ = train_test_split(
            shuffled_index, shuffled_index, test_size=self._test_size
        )
        self._save_sub_df(raw_df, train_index, self._train_df_path, vocab)
        self._save_sub_df(raw_df, val_index, self._val_df_path, vocab)
