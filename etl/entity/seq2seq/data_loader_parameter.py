from pydantic import BaseModel


class DataLoaderParameter(BaseModel):
    train_batch_size: int
    train_n_workers: int
    train_pre_fetch_factor: int
    val_batch_size: int
    val_n_workers: int
    val_pre_fetch_factor: int
    str_max_length: int
