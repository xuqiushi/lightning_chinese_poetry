import torch
from pydantic import BaseModel


class TrainParameter(BaseModel):
    device: torch.device
    epochs: int
    clip: float
    learning_rate: float
    lr_gamma: float

    class Config:
        arbitrary_types_allowed = True
