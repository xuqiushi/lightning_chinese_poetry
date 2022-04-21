import torch
from pydantic import BaseModel


class TransformerModelParameter(BaseModel):
    hid_dim: int = 256
    enc_layers: int = 3
    dec_layers: int = 3
    enc_heads: int = 8
    dec_heads: int = 8
    enc_pf_dim: int = 512
    dec_pf_dim: int = 512
    enc_dropout: float = 0.1
    dec_dropout: float = 0.1
    str_max_length: int
    device: torch.device

    class Config:
        arbitrary_types_allowed = True
