import torch

from etl.dataset.seq2seq_data_loader import Seq2seqDataLoader
from etl.entity.seq2seq.data_loader_parameter import DataLoaderParameter
from etl.entity.seq2seq.data_transformer_parameter import DataTransformerParameter
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence_arrow.raw_data_transformer import RawDataTransformer
from model.base.seq2seq_trainer import Seq2seqTrainer
from model.entity.train_parameter import TrainParameter
from model.entity.transformer_model_parameter import TransformerModelParameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STR_MAX_LENGTH = 100
DATA_TRANSFORMER_PARAMETER = DataTransformerParameter(
    src_directory=TANG_SONG_SHI_DIRECTORY,
    reset_tmp_file=False,
    reset_vocab=False,
    reset_train_val_df=False,
    test_size=0.2,
)
DATA_LOADER_PARAMETER = DataLoaderParameter(
    train_batch_size=384,
    train_n_workers=4,
    train_pre_fetch_factor=8,
    val_batch_size=384,
    val_n_workers=4,
    val_pre_fetch_factor=8,
    str_max_length=STR_MAX_LENGTH,
)
TRANSFORMER_MODEL_PARAMETER = TransformerModelParameter(
    hid_dim=256,
    enc_layers=3,
    dec_layers=3,
    enc_heads=8,
    dec_heads=8,
    enc_pf_dim=512,
    dec_pf_dim=512,
    enc_dropout=0.1,
    dec_dropout=0.1,
    str_max_length=STR_MAX_LENGTH,
    device=device,
)
TRAIN_PARAMETER = TrainParameter(
    device=device,
    epochs=10,
    clip=1,
    learning_rate=0.001,
    lr_gamma=0.8,
)


class Trainer(Seq2seqTrainer):
    pass


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    # torch.backends.cudnn.benchmark = True

    data_transformer = RawDataTransformer(
        seq2seq_data_transformer_parameter=DATA_TRANSFORMER_PARAMETER
    )
    data_loader = Seq2seqDataLoader(
        raw_data_transformer=data_transformer,
        loader_parameter=DATA_LOADER_PARAMETER,
    )
    trainer = Trainer(
        seq2seq_data_loader=data_loader,
        transformer_model_parameter=TRANSFORMER_MODEL_PARAMETER,
        train_parameter=TRAIN_PARAMETER,
    )
    trainer.process()
