from pydantic import BaseModel


class OneSentenceDataInfo(BaseModel):
    src_max_length: int
    tar_max_length: int
    record_count: int
