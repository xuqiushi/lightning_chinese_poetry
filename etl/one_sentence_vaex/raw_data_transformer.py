import vaex

from etl.etl_contants import TANG_SONG_SHI_DIRECTORY


class RawDataTransformer:
    pass


if __name__ == "__main__":
    data = vaex.from_json(str(TANG_SONG_SHI_DIRECTORY / "*.json"), orient='table', copy_index=False)
    raw_data_transformer = RawDataTransformer()
