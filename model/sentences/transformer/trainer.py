import torch

from etl.dataset.seq2seq_data_loader import Seq2seqDataLoader
from etl.entity.seq2seq.data_loader_parameter import DataLoaderParameter
from etl.entity.seq2seq.data_transformer_parameter import DataTransformerParameter
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.sentences.sentences_data_transformer import SentencesDataTransformer
from model.base.seq2seq_trainer import Seq2seqTrainer
from model.entity.train_parameter import TrainParameter
from model.entity.transformer_model_parameter import TransformerModelParameter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STR_MAX_LENGTH = 500
DATA_TRANSFORMER_PARAMETER = DataTransformerParameter(
    src_directory=TANG_SONG_SHI_DIRECTORY,
)
DATA_LOADER_PARAMETER = DataLoaderParameter(
    str_max_length=STR_MAX_LENGTH,
    train_batch_size=256,
    val_batch_size=256,
)
TRANSFORMER_MODEL_PARAMETER = TransformerModelParameter(
    str_max_length=STR_MAX_LENGTH,
    device=DEVICE,
)
TRAIN_PARAMETER = TrainParameter(
    device=DEVICE,
)


class Trainer(Seq2seqTrainer):
    def __init__(self):
        data_transformer = SentencesDataTransformer(
            seq2seq_data_transformer_parameter=DATA_TRANSFORMER_PARAMETER
        )
        data_loader = Seq2seqDataLoader(
            raw_data_transformer=data_transformer,
            loader_parameter=DATA_LOADER_PARAMETER,
        )
        super().__init__(
            seq2seq_data_loader=data_loader,
            transformer_model_parameter=TRANSFORMER_MODEL_PARAMETER,
            train_parameter=TRAIN_PARAMETER,
        )


if __name__ == "__main__":

    trainer = Trainer()
    trainer.process()
