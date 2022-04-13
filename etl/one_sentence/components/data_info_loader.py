import json

from tqdm import tqdm

from etl.entity.one_sentence_data_info import OneSentenceDataInfo
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.base.trainable_component_loader import (
    TrainableComponentLoader,
    ModelType,
)
from etl.one_sentence.components.character_tokenizer import CharacterTokenizer
from etl.one_sentence.custom_dataset import CustomDataset


class DataInfoLoader(TrainableComponentLoader[OneSentenceDataInfo]):
    def __init__(self, dataset: CustomDataset, rebuild: bool = False):
        super(DataInfoLoader, self).__init__(rebuild)
        self.dataset = dataset
        self.tokenizer = CharacterTokenizer()

    @property
    def _model_file_suffix(self) -> str:
        return "json"

    def _load_model(self) -> OneSentenceDataInfo:
        return OneSentenceDataInfo.parse_file(self._model_path)

    def _build_model(self) -> None:
        one_sentence_data_info = OneSentenceDataInfo(
            src_max_length=0,
            tar_max_length=0,
            record_count=0,
        )
        for source, target in tqdm(self.dataset):
            src_len = len(self.tokenizer(source))
            tar_len = len(self.tokenizer(target))
            one_sentence_data_info.src_max_length = max(one_sentence_data_info.src_max_length, src_len)
            one_sentence_data_info.tar_max_length = max(one_sentence_data_info.tar_max_length, tar_len)
            one_sentence_data_info.record_count += 1

        with open(self._model_path, "w") as f:
            json.dump(one_sentence_data_info.dict(), f)


if __name__ == "__main__":
    test_dataset = CustomDataset(TANG_SONG_SHI_DIRECTORY)
    test_data_info_loader = DataInfoLoader(test_dataset)
    test_vocab = test_data_info_loader.load_model()