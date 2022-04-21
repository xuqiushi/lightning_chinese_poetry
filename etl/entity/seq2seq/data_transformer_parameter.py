import pathlib

from pydantic import BaseModel


class DataTransformerParameter(BaseModel):
    src_directory: pathlib.Path
    reset_tmp_file: bool = False
    reset_vocab: bool = False
    reset_train_val_df: bool = False
    test_size: float = 0.2
