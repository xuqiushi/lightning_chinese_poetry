from typing import Tuple

from pydantic import BaseModel


class Poetry(BaseModel):
    id: str
    author: str
    title: str
    paragraphs: Tuple[str, ...]
