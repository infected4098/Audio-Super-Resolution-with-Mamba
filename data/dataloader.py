import numpy as np
from os import path
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from prefetch_generator import BackgroundGenerator
import random
class DataLoader_back(DataLoader):
    def __init__(self, *args, **kwargs):
        super(DataLoader_back, self).__init__(*args, **kwargs)
        if 'num_workers' in kwargs:
            self.num_workers = kwargs['num_workers']
            print('num_workers: ', self.num_workers)
        else:
            self.num_workers = 1

    def __iter__(self):
        return BackgroundGenerator(super().__iter__(),
                                   max_prefetch=self.num_workers // 4)


def create_vctk_dataloader(hparams, cv):
    def collate_fn(batch):
        wav_list = list()
        wav_l_list = list()
        for wav, wav_l in batch:
            wav_list.append(wav)
            wav_l_list.append(wav_l)
        wav_list = torch.stack(wav_list, dim=0).squeeze(1)
        wav_l_list = torch.stack(wav_l_list, dim=0).squeeze(1)

        return wav_list, wav_l_list

    if cv == 0:
        return DataLoader_back(dataset=VCTKMultiSpkDataset(hparams, cv),
                               batch_size=hparams.train.batch_size,
                               shuffle=True,
                               num_workers=hparams.train.num_workers,
                               collate_fn=collate_fn,
                               pin_memory=True,
                               drop_last=True,
                               sampler=None)
    else:
        return DataLoader_back(dataset=VCTKMultiSpkDataset(hparams, cv),
                               collate_fn=collate_fn,
                               batch_size=hparams.train.batch_size if cv == 1 else 1,
                               drop_last=True if cv == 1 else False,
                               shuffle=False,
                               num_workers=hparams.train.num_workers)