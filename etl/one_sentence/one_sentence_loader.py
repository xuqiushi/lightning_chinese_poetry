import pathlib
from typing import Tuple

import torchtext.transforms as torch_transform
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from etl.entity.dataset_type import DatasetType
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.character_tokenizer import CharacterTokenizer
from etl.one_sentence.components.data_info_loader import DataInfoLoader
from etl.one_sentence.components.vocab_loader import BOS, EOS, VocabLoader, PADDING
from etl.one_sentence.custom_dataset import CustomDataset


class OneSentenceLoader:
    def __init__(self, directory: pathlib.Path, device):
        self.device = device
        self.vocab = VocabLoader(CustomDataset(directory=directory)).load_model()
        self.data_info = DataInfoLoader(CustomDataset(directory=directory)).load_model()
        self.t_sequential = torch_transform.Sequential(
            CharacterTokenizer(),
            torch_transform.VocabTransform(self.vocab),
            torch_transform.AddToken(token=self.vocab[BOS], begin=True),
            torch_transform.AddToken(token=self.vocab[EOS], begin=False),
            torch_transform.ToTensor(PADDING),
        )
        self.train_dataset = CustomDataset(
            directory=directory, dataset_type=DatasetType.TRAIN
        )
        self.train_dataset = IterableWrapper(self.train_dataset).map(self.transform)
        self.test_dataset = CustomDataset(
            directory=directory, dataset_type=DatasetType.TEST
        )
        self.test_dataset = IterableWrapper(self.test_dataset).map(self.transform)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=32,
            collate_fn=self.pad_collate,
            num_workers=6,
            pin_memory=True,
            prefetch_factor=16,
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1024, collate_fn=self.pad_collate
        )

    def transform(self, iter_item: Tuple[str, str]):
        return self.t_sequential(iter_item[0]), self.t_sequential(iter_item[1])

    def pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        x_lens = [len(x) for x in xx]
        y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)

        return xx_pad.to(self.device), yy_pad.to(self.device)
