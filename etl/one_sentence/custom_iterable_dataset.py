import logging
import math
import re
from typing import List, Optional, Generator, Tuple
import os
import pathlib
import json

import torch
from torch.utils.data import IterableDataset

from etl.entity.dataset_type import DatasetType
from etl.entity.poetry import Poetry
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY


class CustomIterableDataset(IterableDataset):
    CH_SEP = ",，.。!！?？"
    LOGGER = logging.getLogger("OneSentenceDataset")

    def __init__(
        self,
        directory: pathlib.Path,
        start: Optional[int] = None,
        end: Optional[int] = None,
        dataset_type: DatasetType = DatasetType.ALL,
        train_test_ratio: float = 0.7,
    ):
        super().__init__()
        self.file_paths = self.get_file_paths(directory)
        self.start = start if start else 0
        self.end = end if end else len(self.file_paths)
        self.dataset_type = dataset_type
        self.train_test_ratio = train_test_ratio
        assert self.end > self.start, "end must larger than start"
        assert self.end <= len(
            self.file_paths
        ), "end must smaller than file_paths length"

    @classmethod
    def get_file_paths(
        cls, directory: pathlib.Path, sort_key_regex: str = r"\.(\d+).json"
    ) -> List[str]:
        paths = []
        for dir_path, dir_names, filenames in os.walk(directory):
            for filename in filenames:
                paths.append(pathlib.Path(dir_path) / filename)
        return sorted(
            list(paths),
            key=lambda x: int(re.search(sort_key_regex, str(x)).group(1)),
        )

    def __iter__(self) -> Generator[Tuple[str, str], None, None]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        for file_index in range(iter_start, iter_end):
            with open(self.file_paths[file_index], "r") as f:
                current_json = json.load(f)
                for item_index, item in enumerate(current_json):
                    if self.dataset_type == DatasetType.TRAIN:
                        if item_index % 10 >= self.train_test_ratio * 10:
                            continue
                    if self.dataset_type == DatasetType.TEST:
                        if item_index % 10 < self.train_test_ratio * 10:
                            continue
                    poetry = Poetry.parse_obj(item)
                    for paragraph in poetry.paragraphs:
                        re.fullmatch(
                            rf"^[^{self.CH_SEP}]+[{self.CH_SEP}][^{self.CH_SEP}]+[{self.CH_SEP}]$",
                            paragraph,
                        )
                        sentences = re.search(
                            rf"^([^{self.CH_SEP}]+[{self.CH_SEP}])([^{self.CH_SEP}]+[{self.CH_SEP}])$",
                            paragraph,
                        )
                        if sentences:
                            sentences_groups = sentences.groups()
                            yield sentences_groups
                        else:
                            self.LOGGER.info(
                                f"current paragraph not combine by two sentence: {paragraph}"
                            )


if __name__ == "__main__":
    test_dataset = CustomIterableDataset(TANG_SONG_SHI_DIRECTORY)
    test_one_worker = iter(
        torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1)
    )
    print(next(test_one_worker))
    print(next(test_one_worker))
    test_two_worker = iter(torch.utils.data.DataLoader(test_dataset, num_workers=2))
    print(next(test_two_worker))
    print(next(test_two_worker))
