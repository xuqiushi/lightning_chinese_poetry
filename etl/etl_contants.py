import pathlib

from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger

from config import config

ETL_ROOT = pathlib.Path(__file__).parent.absolute()
TANG_SONG_SHI_DIRECTORY = DirectoryChanger.get_new_root_directory(
    ETL_ROOT / "raw" / "tang_song_shi",
    config.directories.base_dir,
    config.directories.data_dir,
)
CH_SEP = ",，.。!！?？"
PADDING_IDX = 1
BOS_IDX = 0
EOS_IDX = 2
MAX_SEQ_LEN = 256
UNKNOWN = "<unk>"
BOS = "<bos>"
PADDING = "<padding>"
EOS = "<eos>"