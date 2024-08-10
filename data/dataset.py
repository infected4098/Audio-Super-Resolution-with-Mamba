import numpy as np
import os
import math
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
import utils
import filter
from prefetch_generator import BackgroundGenerator
from librosa.util import normalize
import hparams
from librosa.filters import mel as librosa_mel_fn
import random


def worker_init_fn(worker_id, num_workers, rank, seed):
    # Set the worker seed to num_workers * rank + worker_id + seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class VCTKMultiSpkDataset(Dataset):
    def __init__(self, hparams, cv=0, shuffle = True): # cv 0: train, 1: val

        self.cv = cv
        random.seed(1229)

        #filenames
        self.mother_dir = hparams.data["dir"]
        self.train_data_files = hparams.data["train_names"]
        self.val_data_files = hparams.data["val_names"]

        #preprocessing
        self.UPR = hparams.audio["UPR"]
        self.segment_size = hparams.audio["segment_size"]
        self.shuffle = shuffle

        if self.cv == 0:
            audio_files = self.train_data_files
        else:
            audio_files = self.val_data_files
        with open(audio_files, "r") as f:
            self.files = f.readlines()
        if self.shuffle:
            random.shuffle(self.files)

        self.files = [os.path.join(self.mother_dir, utils.preprocess_path(filepath)) for filepath in self.files if os.path.exists(os.path.join(self.mother_dir, utils.preprocess_path(filepath)))]
        # 13 files are missing. Total 40690 training data, 3352 validation data
    def __getitem__(self, index):

        filepath = self.files[index]
        #filepath = os.path.join(self.mother_dir, utils.preprocess_path(filepath))
        upsampling_ratio = random.choice(self.UPR)
        y, sr = utils.load_audio(filepath, sr = 48000)
        y = normalize(y) * 0.95
        sr_low_calc, cutoff_freq = utils.calc_sr_by_UPR(sr, upsampling_ratio)
        y_low, sr_low = filter.build_lowres(y, sr, cutoff_freq)  # assert sr_low_calc = cutoff_freq
        y_rebuilt, sr_rebuilt = y_low, sr_low
        #y_rebuilt, sr_rebuilt = filter.resample_audio(y_low, sr_low, sr)


        if y.shape != y_rebuilt.shape:
            raise ValueError("{} rebuilt wav doesn't match target {} wav".format(y.shape, y_rebuilt.shape))
        
        y_high = torch.FloatTensor(y).unsqueeze(0)
        y_low = torch.FloatTensor(y_rebuilt).unsqueeze(0)

        if y_high.size(1) >= self.segment_size:

            try:
                trim = self.segment_size//4 #trim audio silence
                max_audio_start = y_high.size(1) - self.segment_size - trim
                audio_start = random.randint(trim, max_audio_start)
            except:
                max_audio_start = y_high.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)


            y_high = y_high[:, audio_start:audio_start + self.segment_size]
            y_low = y_low[:, audio_start:audio_start + self.segment_size]

        else:
            y_high = torch.nn.functional.pad(y_high, (0, self.segment_size - y_high.size(1)), 'constant')
            y_low = torch.nn.functional.pad(y_low, (0, self.segment_size - y_high.size(1)), "constant")

        return y_high, y_low, filepath

    def __len__(self):
        return len(self.files)


"""import json
from env import AttrDict
with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:

    json_config = json.load(file)
    hyperparameters = AttrDict(json_config)
    dataset = VCTKMultiSpkDataset(hparams = hyperparameters, cv= 0)
    y_high, y_low, filepath = dataset.__getitem__(8)
    print("length of the dataset is ", dataset.__len__(), "\n", y_high.shape, y_low.shape, filepath)
    a = 1+2"""