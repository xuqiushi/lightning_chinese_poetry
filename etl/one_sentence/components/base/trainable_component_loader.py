from typing import TypeVar, Generic
from abc import ABCMeta, abstractmethod
import pathlib
from lightning_fast.tools.path_tools.class_directory import ClassDirectory
from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger

from config import config

ModelType = TypeVar("ModelType")


class TrainableComponentLoader(Generic[ModelType], metaclass=ABCMeta):
    def __init__(self, rebuild: bool = False):
        self.rebuild = rebuild

    @property
    @abstractmethod
    def _model_file_suffix(self) -> str:
        pass

    @abstractmethod
    def _load_model(self) -> ModelType:
        pass

    @abstractmethod
    def _build_model(self) -> None:
        pass

    @property
    def current_class_path(self) -> pathlib.Path:
        return ClassDirectory(self.__class__).get_class_path()

    @property
    def _need_build(self) -> bool:
        return self.rebuild or not self._model_path.exists()

    @property
    def _tmp_train_text_path(self):
        directory = DirectoryChanger.get_new_root_directory(
            self.current_class_path,
            config.directories.base_dir,
            config.directories.tmp_dir,
        )
        return directory / "tmp_train_text.txt"

    @property
    def _model_directory(self) -> pathlib.Path:
        return DirectoryChanger.get_new_root_directory(
            origin_path=self.current_class_path,
            source_dir=config.directories.base_dir,
            target_dir=config.directories.data_dir,
        )

    @property
    def _model_path(self) -> pathlib.Path:
        return self._model_directory / f"{self.__class__.__name__}.{self._model_file_suffix}"

    def load_model(self) -> ModelType:
        if self._need_build:
            self._build_model()
        return self._load_model()
