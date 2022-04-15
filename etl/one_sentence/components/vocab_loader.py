from typing import Generator

import torch
from torchtext.vocab import build_vocab_from_iterator, Vocab
from tqdm import tqdm

from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.base.trainable_component_loader import (
    TrainableComponentLoader,
)

from etl.one_sentence.components.character_tokenizer import CharacterTokenizer
from etl.one_sentence.custom_iterable_dataset import CustomIterableDataset

UNKNOWN = "<unk>"
BOS = "<bos>"
PADDING = "<padding>"
EOS = "<eos>"


class VocabLoader(TrainableComponentLoader[Vocab]):
    def __init__(self, dataset: CustomIterableDataset, rebuild: bool = False):
        super(VocabLoader, self).__init__(rebuild)
        self.dataset = dataset
        self.tokenizer = CharacterTokenizer()

    @property
    def _model_file_suffix(self) -> str:
        return "pt"

    def _load_model(self) -> Vocab:
        return torch.load(self._model_path)

    def _build_model(self) -> None:
        def _vocab_iter() -> Generator[str, None, None]:
            for source, target in self.dataset:
                yield self.tokenizer(source)
                yield self.tokenizer(target)

        vocab = build_vocab_from_iterator(
            tqdm(_vocab_iter()),
            specials=[UNKNOWN, BOS, PADDING, EOS],
            special_first=True,
        )
        vocab.set_default_index(vocab[UNKNOWN])
        torch.save(vocab, self._model_path)


if __name__ == "__main__":
    test_dataset = CustomIterableDataset(TANG_SONG_SHI_DIRECTORY)
    test_vocab_builder = VocabLoader(test_dataset, rebuild=True)
    test_vocab = test_vocab_builder.load_model()
