from __future__ import absolute_import, division, print_function, unicode_literals
import utils
import glob
import os
import numpy as np
from librosa.util import normalize
import wandb
import filter
import random
import argparse
import librosa
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from modules.mms import asmr_full as Generator
from datetime import timedelta
from tqdm import tqdm
h = None

"""torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://localhost:12345',
                                         world_size=4,
                                         rank=0, timeout=timedelta(minutes=30))"""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"
    print("setup initialized...")
def init_process(rank):
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='tcp://localhost:12345',
                                         world_size=4,
                                         rank=rank, timeout=timedelta(minutes=30))
    #torch.distributed.barrier()

def prefix_load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    prefix = "module."
    for key in state_dict["generator"].keys():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # Remove the prefix
            new_state_dict[new_key] = state_dict["generator"][key]

    print("Complete.")
    return new_state_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:
    json_config = json.load(file)
    cfg = AttrDict(json_config)


ckpt_path = "/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/ckpts_full_l1/g_00002000"
val_names = "/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/dataset_names/val-files.txt"
output_dir = cfg.data["etc_save_path"]
device = torch.device("cuda")
#checkpoint = prefix_load_checkpoint(ckpt_path, device)

#missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)

#state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint


def inference(a, count = 10):
    global device
    setup(0, 8)
    #init_process(0)
    wandb.init(project="inference_asmr_mel_trim_l1_small")
    wandb.require("core")
    random.seed(cfg.info["seed"])
    #global cfg
    print("wandb initialized...")
    generator = Generator(cfg.embed, cfg.hifigan, cfg.mamba).to(device)
    print("model initialized...")
    mother_dir = cfg.data["dir"]
    val_data_files = cfg.data["val_names"]
    UPR = cfg.audio["UPR"]
    state_dict_g = prefix_load_checkpoint(a.checkpoint_file, {'cuda:%d' % 0: 'cuda:%d' % 3})
    generator.load_state_dict(state_dict_g, strict= True)

    with open(val_data_files, "r") as f:
        files = f.readlines()

    os.makedirs(a.output_dir, exist_ok=True)
    files = [os.path.join(mother_dir, utils.preprocess_path(filepath)) for filepath in files if
             os.path.exists(os.path.join(mother_dir, utils.preprocess_path(filepath)))]
    generator.eval()
    generator.remove_weight_norm()

    files = random.sample(files, count)

    with torch.no_grad():
        for i, filename in tqdm(enumerate(files)):

            y_high, sr_high = utils.load_audio(filename, sr = 48000)
            y_high = utils.trim_or_pad(y_high, 128)
            y_high = normalize(y_high) * 0.95
            gt_high_mel = librosa.power_to_db(
                utils.mel_spectrogram(y_high, cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                cfg.audio["hop"],
                                cfg.audio["window_size"], cfg.audio["fmin"],
                                cfg.audio["fmax_for_loss"]), ref=np.max)
            upsampling_ratio = random.choice(UPR)
            _, cutoff_freq = utils.calc_sr_by_UPR(sr_high, upsampling_ratio)
            gt_high_audio = wandb.Audio(y_high, sample_rate=cfg.audio["sr"], caption="ground_truth_high_audio")
            y_low, sr_low = filter.build_lowres(y_high, sr_high, cutoff_freq)
            gt_low_mel = librosa.power_to_db(
                utils.mel_spectrogram(y_low, cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                      cfg.audio["hop"],
                                      cfg.audio["window_size"], cfg.audio["fmin"],
                                      cfg.audio["fmax_for_loss"]), ref=np.max)
            gt_low_audio = wandb.Audio(y_low, sample_rate=cfg.audio["sr"], caption="ground_truth_low_audio")
            y_low = torch.FloatTensor(y_low).to(device)


            y_g_hat = generator(y_low.unsqueeze(0).unsqueeze(0))
            audio = y_g_hat.squeeze()
            audio = audio.cpu().numpy()
            pred_mel = librosa.power_to_db(
                utils.mel_spectrogram(audio, cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                cfg.audio["hop"],
                                cfg.audio["window_size"], cfg.audio["fmin"],
                                cfg.audio["fmax_for_loss"]), ref=np.max)
            pred_audio = wandb.Audio(audio, sample_rate=cfg.audio["sr"], caption="predicted_audio")
            gt_high_mel = wandb.Image(
                librosa.display.specshow(gt_high_mel, sr=cfg.audio["sr"], x_axis='time', y_axis='mel',
                                         fmax=24000), caption="gt_high_mel")
            gt_low_mel = wandb.Image(librosa.display.specshow(gt_low_mel, sr=cfg.audio["sr"], x_axis='time',
                                                              y_axis='mel', fmax=24000),
                                     caption="gt_low_mel")
            pred_mel = wandb.Image(librosa.display.specshow(pred_mel, sr=cfg.audio["sr"], x_axis='time',
                                                            y_axis='mel', fmax=24000),
                                   caption="predicted_mel")

            wandb.log({f"ground truth of {i} th high mel": gt_high_mel})
            wandb.log({f"ground truth of {i} th low mel": gt_low_mel})
            wandb.log({f"predicted {i} th mel": pred_mel})
            wandb.log({f"ground truth of {i} th high audio": gt_high_audio})
            wandb.log({f"ground truth of {i} th low audio": gt_low_audio})
            wandb.log({f"predicted {i} th audio": pred_audio})
            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated_e2e.wav')
            write(output_file, cfg.audio["sr"], audio)


            print(output_file)
            print("inference finished...")



def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--validation_file', default=val_names)
    parser.add_argument('--output_dir', default='generated_files_from_mel')
    parser.add_argument('--checkpoint_file', default = ckpt_path)
    a = parser.parse_args()

    with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:

        json_config = json.load(file)
        h = AttrDict(json_config)
        print("config initialized..")
    torch.manual_seed(h.info["seed"])
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.info["seed"])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    inference(a)


if __name__ == '__main__':
    main()
