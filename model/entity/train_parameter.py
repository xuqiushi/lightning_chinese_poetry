import torch
from pydantic import BaseModel


class TrainParameter(BaseModel):
    device: torch.device
    epochs: int = 10
    clip: float = 1
    learning_rate: float = 0.00005
    lr_gamma: float = 0.8

    class Config:
        arbitrary_types_allowed = True
