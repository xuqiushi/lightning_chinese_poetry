import pathlib
from typing import Optional

from lightning_fast.tools.luigi_tools.task_base import TaskBase

from config import config


class LuigiBase(TaskBase):
    @property
    def base_dir(self) -> Optional[pathlib.Path]:
        return config.directories.base_dir

    @property
    def data_dir(self) -> Optional[pathlib.Path]:
        return config.directories.data_dir
