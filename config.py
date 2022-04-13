import pathlib

from lightning_fast.config_factory import Config, SimpleDirectoryCollection


BASE_DIRECTORY = pathlib.Path(__file__).absolute().parent


config = Config(
    config_path=BASE_DIRECTORY / "settings.yaml",
    directories=SimpleDirectoryCollection(base_directory=BASE_DIRECTORY),
)
