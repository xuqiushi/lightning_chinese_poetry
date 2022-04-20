from pydantic import BaseModel


class DataLoaderParameter(BaseModel):
    batch_size: int
    n_workers: int
    pre_fetch_factor: int
