from typing import Tuple, List

import torch
import torchtext.transforms as torch_transform
from pyarrow import Table
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper

from etl.base.base_seq2seq_data_transformer import BaseSeq2seqDataTransformer
from etl.dataset.arrow_dataset import ArrowDataset
from etl.entity.data_loader_parameter import DataLoaderParameter
from etl.etl_contants import PADDING


class Seq2seqDataLoader:
    def __init__(
        self,
        raw_data_transformer: BaseSeq2seqDataTransformer,
        train_loader_parameter: DataLoaderParameter,
        val_loader_parameter: DataLoaderParameter,
        device: torch.device,
        str_max_length: int = 100,
    ):
        self._device = device
        self._raw_data_transformer = raw_data_transformer
        self.vocab = self._raw_data_transformer.get_vocab()
        self._t_sequential = torch_transform.Sequential(
            torch_transform.Truncate(str_max_length),
            torch_transform.ToTensor(self.vocab[PADDING]),
        )
        train_df, val_df = self._raw_data_transformer.get_train_test()
        self.train_loader = self._get_data_loader(train_df, train_loader_parameter)
        self.val_loader = self._get_data_loader(val_df, val_loader_parameter)

    def _pad_collate(self, batch):
        (xx, yy) = zip(*batch)
        # x_lens = [len(x) for x in xx]
        # y_lens = [len(y) for y in yy]

        xx_pad = pad_sequence(xx, batch_first=True, padding_value=self.vocab[PADDING])
        yy_pad = pad_sequence(yy, batch_first=True, padding_value=self.vocab[PADDING])

        return xx_pad, yy_pad

    def _transform(self, iter_item: Tuple[List[int], List[int]]):
        return self._t_sequential(iter_item[0]), self._t_sequential(iter_item[1])

    def _get_data_loader(
        self, df: Table, data_loader_parameter: DataLoaderParameter
    ) -> DataLoader:
        dataset = ArrowDataset(df)
        dataset = SequenceWrapper(dataset).map(self._transform)
        return DataLoader(
            dataset,
            batch_size=data_loader_parameter.batch_size,
            collate_fn=self._pad_collate,
            num_workers=data_loader_parameter.n_workers,
            pin_memory=True,
            prefetch_factor=data_loader_parameter.pre_fetch_factor,
        )
