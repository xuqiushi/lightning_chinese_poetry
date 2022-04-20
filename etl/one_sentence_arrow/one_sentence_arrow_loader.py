import pathlib
from typing import Tuple

import torch
import torchtext.transforms as torch_transform
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper

from etl.etl_contants import PADDING
from etl.one_sentence_arrow.one_sentence_arrow_dataset import OneSentenceArrowDataset
from etl.one_sentence_arrow.raw_data_transformer import RawDataTransformer


class OneSentenceArrowLoader:
    def __init__(
        self,
        directory: pathlib.Path,
        train_batch_size: int,
        train_n_workers: int,
        train_pre_fetch_factor: int,
        val_batch_size: int,
        val_n_workers: int,
        val_pre_fetch_factor: int,
        device: torch.device,
        test_size: float = 0.2,
        str_max_length: int = 100,
    ):
        self.device = device
        self.raw_data_transformer = RawDataTransformer(directory, test_size=test_size)
        self.vocab = self.raw_data_transformer.get_vocab()
        self.t_sequential = torch_transform.Sequential(
            torch_transform.Truncate(str_max_length),
            torch_transform.ToTensor(PADDING),
        )
        train_df, val_df = self.raw_data_transformer.get_train_test()
        self.train_dataset = OneSentenceArrowDataset(train_df)
        self.train_dataset = SequenceWrapper(self.train_dataset).map(self.transform)
        self.val_dataset = OneSentenceArrowDataset(val_df)
        self.val_dataset = SequenceWrapper(self.val_dataset).map(self.transform)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            collate_fn=self.pad_collate,
            num_workers=train_n_workers,
            pin_memory=True,
            prefetch_factor=train_pre_fetch_factor,
        )
        self.train_record_count = len(train_df)
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            num_workers=val_n_workers,
            prefetch_factor=val_pre_fetch_factor,
            collate_fn=self.pad_collate,
        )
        self.val_record_count = len(val_df)

    def transform(self, iter_item: Tuple[str, str]):
        return self.t_sequential(iter_item[0]), self.t_sequential(iter_item[1])

    def pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        # x_lens = [len(x) for x in xx]
        # y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

        return xx_pad, yy_pad
