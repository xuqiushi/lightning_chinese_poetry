import pathlib
from typing import Tuple

import torch
import torchtext.transforms as torch_transform
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from etl.entity.dataset_type import DatasetType
from etl.one_sentence.components.character_tokenizer import CharacterTokenizer
from etl.one_sentence.components.data_info_loader import DataInfoLoader
from etl.one_sentence.components.vocab_loader import BOS, EOS, VocabLoader, PADDING
from etl.one_sentence.custom_iterable_dataset import CustomIterableDataset


class OneSentenceLoader:
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
        train_test_ratio: float = 0.7,
    ):
        self.device = device
        self.vocab = VocabLoader(
            CustomIterableDataset(directory=directory)
        ).load_model()
        self.data_info = DataInfoLoader(
            CustomIterableDataset(directory=directory)
        ).load_model()
        self.t_sequential = torch_transform.Sequential(
            CharacterTokenizer(),
            torch_transform.VocabTransform(self.vocab),
            torch_transform.AddToken(token=self.vocab[BOS], begin=True),
            torch_transform.AddToken(token=self.vocab[EOS], begin=False),
            torch_transform.ToTensor(PADDING),
        )
        self.train_dataset = CustomIterableDataset(
            directory=directory,
            dataset_type=DatasetType.TRAIN,
            train_test_ratio=train_test_ratio,
        )
        self.train_dataset = IterableWrapper(self.train_dataset).map(self.transform)
        self.val_dataset = CustomIterableDataset(
            directory=directory,
            dataset_type=DatasetType.TEST,
            train_test_ratio=train_test_ratio,
        )
        self.val_dataset = IterableWrapper(self.val_dataset).map(self.transform)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=train_batch_size,
            collate_fn=self.pad_collate,
            num_workers=train_n_workers,
            pin_memory=True,
            prefetch_factor=train_pre_fetch_factor,
        )
        self.train_record_count = self.data_info.record_count * train_test_ratio
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            num_workers=val_n_workers,
            prefetch_factor=val_pre_fetch_factor,
            collate_fn=self.pad_collate,
        )
        self.val_record_count = self.data_info.record_count * (1 - train_test_ratio)

    def transform(self, iter_item: Tuple[str, str]):
        return self.t_sequential(iter_item[0]), self.t_sequential(iter_item[1])

    def pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        # x_lens = [len(x) for x in xx]
        # y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

        return xx_pad.to(self.device), yy_pad.to(self.device)
