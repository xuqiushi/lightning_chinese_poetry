import math
import pathlib
import time

import torch
from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger
from torch import nn, autocast
from torch.cuda.amp import GradScaler
from torch.nn.modules.loss import CrossEntropyLoss

# from torch.profiler import profile, record_function, ProfilerActivity
from tqdm import tqdm

from config import config
from etl.dataset.seq2seq_data_loader import Seq2seqDataLoader
from etl.entity.data_loader_parameter import DataLoaderParameter
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY, PADDING
from etl.one_sentence_arrow.raw_data_transformer import RawDataTransformer
from model.one_sentence.transformer.net.decoder import Decoder
from model.one_sentence.transformer.net.encoder import Encoder
from model.one_sentence.transformer.net.seq2seq import Seq2Seq

LOG_DIR = config.directories.log_dir / "one_sentence_transformer"
STR_MAX_LENGTH = 100
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
TEST_SIZE = 0.2
# LEARNING_RATE = 0.0005
LEARNING_RATE = 0.0005
LR_GAMMA = 0.9
CLIP = 1
BATCH_SIZE = 384
EPOCHS = 10

TRAIN_LOADER_PARAMETER = DataLoaderParameter(
    batch_size=BATCH_SIZE,
    n_workers=4,
    pre_fetch_factor=8,
)

VAL_LOADER_PARAMETER = DataLoaderParameter(
    batch_size=BATCH_SIZE,
    n_workers=4,
    pre_fetch_factor=8,
)


class Trainer:
    def __init__(self, data_directory: pathlib.Path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_directory = data_directory
        raw_data_transformer = RawDataTransformer(data_directory, test_size=TEST_SIZE)
        self.data_loader = Seq2seqDataLoader(
            raw_data_transformer,
            TRAIN_LOADER_PARAMETER,
            VAL_LOADER_PARAMETER,
            self.device,
            STR_MAX_LENGTH,
        )

        self._vocab = self.data_loader.vocab
        self._src_dim = len(self._vocab)
        self._trg_dim = len(self._vocab)
        self._src_pad_idx = self._vocab[PADDING]
        self._trg_pad_idx = self._vocab[PADDING]

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.criterion = None
        self.scaler = None

    @classmethod
    def count_parameters(cls, model):
        parameters_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"The model has {parameters_count:,} trainable parameters")
        return parameters_count

    @classmethod
    def initialize_weights(cls, model):
        if hasattr(model, "weight") and model.weight.dim() > 1:
            nn.init.xavier_uniform_(model.weight.data)

    @classmethod
    def epoch_time(cls, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_minutes = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_minutes * 60))
        return elapsed_minutes, elapsed_secs

    @property
    def model_path(self):
        return (
            DirectoryChanger.get_new_root_directory(
                pathlib.Path(__file__).absolute(),
                config.directories.base_dir,
                config.directories.data_dir,
            )
            / "seq2seq.pt"
        )

    def init_model(self):
        enc = Encoder(
            self._src_dim,
            HID_DIM,
            ENC_LAYERS,
            ENC_HEADS,
            ENC_PF_DIM,
            ENC_DROPOUT,
            self.device,
            STR_MAX_LENGTH,
        )
        dec = Decoder(
            self._trg_dim,
            HID_DIM,
            DEC_LAYERS,
            DEC_HEADS,
            DEC_PF_DIM,
            DEC_DROPOUT,
            self.device,
            STR_MAX_LENGTH,
        )
        self.model = Seq2Seq(
            enc, dec, self._src_pad_idx, self._trg_pad_idx, self.device
        ).to(self.device)
        self.count_parameters(self.model)
        self.model.apply(self.initialize_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=LR_GAMMA
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=self._trg_pad_idx)
        self.scaler = GradScaler()

    def process(self):
        self.init_model()
        best_valid_loss = float("inf")
        for epoch in range(EPOCHS):
            start_time = time.time()
            train_loss = self.train(
                self.model,
                self.data_loader,
                self.optimizer,
                self.lr_scheduler,
                self.criterion,
                self.device,
                self.scaler,
            )
            valid_loss = self.evaluate(
                self.model, self.data_loader, self.criterion, self.device
            )

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), str(self.model_path))

            print(
                f"Epoch: {epoch + 1:02} | "
                f"Time: {epoch_mins}m {epoch_secs}s | "
                f"Next LR: {self.optimizer.param_groups[0]['lr']:6f}"
            )
            print()
            print(
                f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
            )
            print(
                f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}"
            )

    @classmethod
    def train(
        cls,
        model: Seq2Seq,
        data_loader: Seq2seqDataLoader,
        optimizer: torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler.ExponentialLR,
        criterion: CrossEntropyLoss,
        device: torch.device,
        scaler: GradScaler,
    ):
        model.train()
        epoch_loss = torch.tensor([0.0], device=device)
        scale = scaler.get_scale()
        for i, (src, trg) in enumerate(
            tqdm(
                data_loader.train_loader,
                total=len(data_loader.train_loader),
            )
        ):
            # with profile(
            #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #     record_shapes=True,
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler(str(LOG_DIR))
            # ) as prof:
            #     with record_function("model_inference"):
            # optimizer.zero_grad()
            src = src.to(device, non_blocking=True).long()
            trg = trg.to(device, non_blocking=True).long()
            for param in model.parameters():
                param.grad = None
            with autocast(device.type):
                output, _ = model(src, trg)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)
            loss = criterion(output, trg)
            # loss.backward()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss

            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        skip_lr_sch = scale > scaler.get_scale()
        if not skip_lr_sch:
            lr_scheduler.step()
        epoch_loss = epoch_loss.item()
        return epoch_loss / len(data_loader.train_loader)

    @classmethod
    def evaluate(
        cls,
        model: Seq2Seq,
        data_loader: Seq2seqDataLoader,
        criterion: CrossEntropyLoss,
        device: torch.device,
    ):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (src, trg) in enumerate(
                tqdm(
                    data_loader.val_loader,
                    total=len(data_loader.val_loader),
                )
            ):
                src = src.to(device)
                trg = trg.to(device)
                output, _ = model(src, trg)
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg.contiguous().view(-1)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / len(data_loader.val_loader)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    # torch.backends.cudnn.benchmark = True
    trainer = Trainer(TANG_SONG_SHI_DIRECTORY)
    trainer.process()
