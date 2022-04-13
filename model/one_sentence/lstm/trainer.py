import math
import pathlib
import time

import torch
from torch import nn, optim

from lightning_fast.tools.path_tools.directory_changer import DirectoryChanger
from tqdm import tqdm

from etl.etl_contants import TANG_SONG_SHI_DIRECTORY
from etl.one_sentence.components.vocab_loader import VocabLoader, PADDING
from etl.one_sentence.custom_dataset import CustomDataset
from etl.one_sentence.one_sentence_loader import OneSentenceLoader
from model.one_sentence.lstm.net import Net
from config import config


class Trainer:
    @classmethod
    def init_weights(cls, model):
        for name, param in model.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    @classmethod
    def count_parameters(cls, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @classmethod
    def train(cls, net_model, iterator, optimizer, criterion, clip, train_record_count):
        epoch_loss = 0
        net_model.train()
        count = 0
        for batch_index, (src, tar) in enumerate(
            tqdm(iterator, total=train_record_count)
        ):
            optimizer.zero_grad()
            output = net_model(src, tar)
            output_dim = output.shape[-1]
            output = output[:, 1:, :].reshape(-1, output_dim)
            tar = tar[:, 1:].reshape(-1)
            loss = criterion(output, tar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net_model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
            count += 1
        return epoch_loss / count

    @classmethod
    def evaluate(cls, net_model, iterator, criterion):
        net_model.eval()

        epoch_loss = 0

        count = 0
        with torch.no_grad():
            for i, (src, tar) in enumerate(tqdm(iterator, total=1500)):

                output = net_model(src, tar, 0)  # turn off teacher forcing

                output_dim = output.shape[-1]

                output = output[:, 1:, :].reshape(-1, output_dim)
                tar = tar[:, 1:].reshape(-1)

                loss = criterion(output, tar)

                epoch_loss += loss.item()
                count += 1

        return epoch_loss / count

    @classmethod
    def epoch_time(cls, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = VocabLoader(CustomDataset(TANG_SONG_SHI_DIRECTORY)).load_model()
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = len(vocab)
    ENC_EMB_DIM = 256
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    N_EPOCHS = 10
    CLIP = 1
    test_model = Net(
        input_dim=INPUT_DIM,
        encode_emb_dim=ENC_EMB_DIM,
        encode_dropout=ENC_DROPOUT,
        output_dim=OUTPUT_DIM,
        decode_emb_dim=DEC_EMB_DIM,
        decode_dropout=DEC_DROPOUT,
        hid_dim=HID_DIM,
        n_layers=N_LAYERS,
        device=device,
    ).to(device)
    test_data_loader = OneSentenceLoader(TANG_SONG_SHI_DIRECTORY, device=device)
    test_model.apply(Trainer.init_weights)
    print(
        f"The model has {Trainer.count_parameters(test_model):,} trainable parameters"
    )
    test_optimizer = optim.Adam(test_model.parameters())
    trg_pad_idx = test_data_loader.vocab[PADDING]
    test_criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
    best_valid_loss = float("inf")
    test_train_record_count = test_data_loader.data_info.record_count
    for epoch in range(N_EPOCHS):

        test_start_time = time.time()

        test_train_loss = Trainer.train(
            test_model,
            test_data_loader.train_loader,
            test_optimizer,
            test_criterion,
            CLIP,
            test_train_record_count / 128,
        )
        test_valid_loss = Trainer.evaluate(
            test_model, test_data_loader.test_loader, test_criterion
        )

        test_end_time = time.time()

        epoch_minutes, epoch_secs = Trainer.epoch_time(test_start_time, test_end_time)

        if test_valid_loss < best_valid_loss:
            best_valid_loss = test_valid_loss
            str(
                torch.save(
                    test_model.state_dict(),
                    DirectoryChanger.get_new_root_directory(
                        pathlib.Path(__file__).absolute(),
                        config.directories.base_dir,
                        config.directories.data_dir,
                    )
                    / "tut1-model.pt",
                )
            )

        print(f"Epoch: {epoch + 1:02} | Time: {epoch_minutes}m {epoch_secs}s")
        print(
            f"\tTrain Loss: {test_train_loss:.3f} | Train PPL: {math.exp(test_train_loss):7.3f}"
        )
        print(
            f"\t Val. Loss: {test_valid_loss:.3f} |  Val. PPL: {math.exp(test_valid_loss):7.3f}"
        )
