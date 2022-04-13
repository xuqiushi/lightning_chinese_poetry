import sentencepiece as spm
from torchtext.transforms import SentencePieceTokenizer
from tqdm import tqdm
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.base.trainable_component_loader import (
    TrainableComponentLoader,
    ModelType,
)
from etl.one_sentence.custom_dataset import CustomDataset


class SentencePieceLoader(TrainableComponentLoader[any]):
    def __init__(self, dataset: CustomDataset, rebuild: bool = False):
        super(SentencePieceLoader, self).__init__(rebuild)
        self.dataset = dataset
        self.rebuild = rebuild

    def load_tokenizer(self):
        return SentencePieceTokenizer(sp_model_path=str(self._model_path))

    def _build_train_text(self):
        with open(self._tmp_train_text_path, "w") as f:
            for source, target in tqdm(self.dataset):
                f.write(source)
                f.write(" ")
                f.write(target)
                f.write("\n")
            f.flush()

    @property
    def _model_file_suffix(self) -> str:
        return "model"

    def _load_model(self) -> ModelType:
        # noinspection PyArgumentList
        return spm.SentencePieceProcessor(model_file=str(self._model_path))

    def _build_model(self) -> None:
        self._build_train_text()
        spm.SentencePieceTrainer.Train(
            input=str(self._tmp_train_text_path),
            vocab_size=20000,
            model_prefix=str(self._model_path).rstrip(".model"),
            character_coverage=1.0,
        )


if __name__ == "__main__":
    test_dataset = CustomDataset(TANG_SONG_SHI_DIRECTORY)
    test_sp_builder = SentencePieceLoader(test_dataset, rebuild=False)
    test_sp = test_sp_builder.load_tokenizer()
    print(test_sp("欲出未出光辣達，千山萬山如火發。"))
