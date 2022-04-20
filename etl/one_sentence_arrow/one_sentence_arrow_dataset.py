import pyarrow as pa
from torch.utils.data import Dataset

from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence_arrow.raw_data_transformer import RawDataTransformer


class OneSentenceArrowDataset(Dataset):
    def __init__(self, table: pa.Table):
        self.table = table

    def __len__(self):
        return len(self.table)

    def __getitem__(self, item):
        return self.table[0][item].as_py(), self.table[1][item].as_py()


if __name__ == '__main__':
    raw_data_transformer = RawDataTransformer(TANG_SONG_SHI_DIRECTORY)
    test_train_df, test_val_df = raw_data_transformer.get_train_test()
    for x in OneSentenceArrowDataset(test_train_df):
        print(x)

