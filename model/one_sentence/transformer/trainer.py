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
from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.vocab_loader import VocabLoader, PADDING
from etl.one_sentence.custom_iterable_dataset import CustomIterableDataset
from etl.one_sentence.one_sentence_loader import OneSentenceLoader
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
LEARNING_RATE = 0.0005
CLIP = 1
BATCH_SIZE = 512
EPOCHS = 10


class Trainer:
    def __init__(self, data_directory: pathlib.Path):
        self.data_directory = data_directory

        self.vocab = VocabLoader(CustomIterableDataset(data_directory)).load_model()
        self.src_dim = len(self.vocab)
        self.trg_dim = len(self.vocab)
        self.src_pad_idx = self.vocab[PADDING]
        self.trg_pad_idx = self.vocab[PADDING]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = OneSentenceLoader(
            directory=self.data_directory,
            train_n_workers=4,
            train_batch_size=BATCH_SIZE,
            train_pre_fetch_factor=8,
            val_n_workers=4,
            val_batch_size=BATCH_SIZE,
            val_pre_fetch_factor=8,
            device=self.device,
            str_max_length=STR_MAX_LENGTH,
        )

        self.model = None
        self.optimizer = None
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
            self.src_dim,
            HID_DIM,
            ENC_LAYERS,
            ENC_HEADS,
            ENC_PF_DIM,
            ENC_DROPOUT,
            self.device,
            STR_MAX_LENGTH,
        )
        dec = Decoder(
            self.trg_dim,
            HID_DIM,
            DEC_LAYERS,
            DEC_HEADS,
            DEC_PF_DIM,
            DEC_DROPOUT,
            self.device,
            STR_MAX_LENGTH,
        )
        self.model = Seq2Seq(
            enc, dec, self.src_pad_idx, self.trg_pad_idx, self.device
        ).to(self.device)
        self.count_parameters(self.model)
        self.model.apply(self.initialize_weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_idx)
        self.scaler = GradScaler()

    def process(self):
        torch.multiprocessing.set_start_method("spawn")
        self.init_model()
        model = torch.jit.script(self.model).to(self.device)
        best_valid_loss = float("inf")
        for epoch in range(EPOCHS):

            start_time = time.time()

            train_loss = self.train(
                model,
                self.data_loader,
                self.optimizer,
                self.criterion,
                self.device,
                self.scaler,
            )
            valid_loss = self.evaluate(
                model, self.data_loader, self.criterion, self.device
            )

            end_time = time.time()

            epoch_mins, epoch_secs = self.epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), str(self.model_path))

            print(f"Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s")
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
        data_loader: OneSentenceLoader,
        optimizer: torch.optim.Adam,
        criterion: CrossEntropyLoss,
        device: torch.device,
        scaler: GradScaler,
    ):
        model.train()
        epoch_loss = torch.tensor([0.0], device=device)
        for i, (src, trg) in enumerate(
            tqdm(
                data_loader.train_loader,
                total=data_loader.train_record_count / BATCH_SIZE,
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
                output, _ = torch.jit.trace(model, src, trg[:, :-1])

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output, trg)
            # loss.backward()
            scaler.scale(loss).backward()
            # scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss

            # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

        epoch_loss = epoch_loss.item()
        return epoch_loss / data_loader.train_record_count * BATCH_SIZE

    @classmethod
    def evaluate(
        cls,
        model: Seq2Seq,
        data_loader: OneSentenceLoader,
        criterion: CrossEntropyLoss,
        device: torch.device,
    ):
        model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for i, (src, trg) in enumerate(
                tqdm(
                    data_loader.val_loader,
                    total=data_loader.val_record_count / BATCH_SIZE,
                )
            ):
                src = src.to(device)
                trg = trg.to(device)
                output, _ = model(src, trg[:, :-1])
                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)
                loss = criterion(output, trg)
                epoch_loss += loss.item()
        return epoch_loss / data_loader.val_record_count * BATCH_SIZE


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)
    # torch.backends.cudnn.benchmark = True
    trainer = Trainer(TANG_SONG_SHI_DIRECTORY)
    trainer.process()
