from pydantic import BaseModel


class DataLoaderParameter(BaseModel):
    train_batch_size: int = 256
    train_n_workers: int = 4
    train_pre_fetch_factor: int = 8
    val_batch_size: int = 256
    val_n_workers: int = 4
    val_pre_fetch_factor: int = 8
    str_max_length: int
