import pathlib

from pydantic import BaseModel


class DataTransformerParameter(BaseModel):
    src_directory: pathlib.Path
    reset_tmp_file: bool
    reset_vocab: bool
    reset_train_val_df: bool
    test_size: float
