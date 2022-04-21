import torch
from pydantic import BaseModel


class TransformerModelParameter(BaseModel):
    hid_dim: int
    enc_layers: int
    dec_layers: int
    enc_heads: int
    dec_heads: int
    enc_pf_dim: int
    dec_pf_dim: int
    enc_dropout: float
    dec_dropout: float
    str_max_length: int
    device: torch.device

    class Config:
        arbitrary_types_allowed = True
