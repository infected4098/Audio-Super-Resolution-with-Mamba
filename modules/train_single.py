import warnings
from env import AttrDict, build_env
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import itertools
import librosa
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from data.dataset import VCTKMultiSpkDataset
from hparams import hyperparameters
from modules.mms import asmr, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss, \
    discriminator_loss
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, mel_spectrogram
import wandb
torch.backends.cudnn.benchmark = True


def train(rank, cfg):

    #Wandb logging
    wandb.init(project=cfg.info["project_name"])
    wandb.require("core")
    wandb.run.name = ""
    wandb.run.save()
    wandb.config.update(cfg)

    # Early Stopping
    init_err = 99999
    best_err_for_ckpt = 99999
    best_steps = 0
    early_stopping_count = 0

    #setup
    torch.cuda.manual_seed(cfg.info["seed"])
    device = torch.device('cuda:1')

    generator = asmr(cfg.embed, cfg.hifigan).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)
    cp_g, cp_do = None, None
    if rank == 0:
        print(generator)
        os.makedirs(cfg.data["ckpt_save_path"], exist_ok=True)
        print("checkpoints directory : ", cfg.data["ckpt_save_path"])

    if os.path.isdir(cfg.data["ckpt_save_path"]):
        cp_g = scan_checkpoint(cfg.data["ckpt_save_path"], 'g_')
        cp_do = scan_checkpoint(cfg.data["ckpt_save_path"], 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']


    optim_g = torch.optim.AdamW(generator.parameters(), cfg.train["lr"], betas=[cfg.train["beta1"], cfg.train["beta2"]])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                cfg.train["lr"], betas=[cfg.train["beta1"], cfg.train["beta2"]])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=cfg.train["weight_decay"], last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=cfg.train["weight_decay"], last_epoch=last_epoch)

    trainset = VCTKMultiSpkDataset(hparams = cfg) #training
    train_sampler = None

    train_loader = DataLoader(trainset, num_workers=cfg.train["num_workers"], shuffle=False,
                              sampler=train_sampler,
                              batch_size=cfg.train["batch_size"],
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        validset = VCTKMultiSpkDataset(hparams = cfg, cv = 1) #validation
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)


    generator.train()
    mpd.train()
    msd.train()

    for epoch in range(max(0, last_epoch), cfg.train["n_epoch"]):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch + 1))



        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            y_high, y_low, _ = batch
            y_high = torch.autograd.Variable(y_high.to(device, non_blocking=True))
            y_low = torch.autograd.Variable(y_low.to(device, non_blocking=True))


            try:
                y_g_hat = generator(y_low)
            except:
                print(y_low.shape)
                continue

            y_mel = torch.from_numpy(mel_spectrogram(y_high.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"], cfg.audio["hop"],
                                    cfg.audio["window_size"], cfg.audio["fmin"], cfg.audio["fmax_for_loss"]))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y_low_mel = torch.from_numpy(mel_spectrogram(y_low.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"], cfg.audio["hop"],
                                    cfg.audio["window_size"], cfg.audio["fmin"], cfg.audio["fmax_for_loss"]))
            y_low_mel = torch.autograd.Variable(y_low_mel.to(device, non_blocking=True))

            try:

                y_g_hat_mel = torch.from_numpy(mel_spectrogram(y_g_hat.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"], cfg.audio["hop"],
                                       cfg.audio["window_size"], cfg.audio["fmin"], cfg.audio["fmax_for_loss"]))
            except:
                print("error point is: ", y_g_hat_mel.shape)
                nan_mask = torch.isnan(y_g_hat)
                nan_indices = torch.nonzero(nan_mask)
                print("NaN indices:")
                print(nan_indices)
                break


            y_g_hat_mel = torch.autograd.Variable(y_g_hat_mel.to(device, non_blocking=True))
            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y_high, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y_high, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(mpd.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(msd.parameters(), max_norm=1.0)

            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y_high, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y_high, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

            loss_gen_all.backward()
            optim_g.step()


            if steps % cfg.train["stdout_interval"] == 0:
                with torch.no_grad():
                    mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                      format(steps, loss_gen_all, mel_error, time.time() - start_b))


            # wandb summary logging
            if steps % cfg.train["summary_interval"] == 0:
                wandb.log({"training/gen_loss_total": loss_gen_all, "steps": steps})
                wandb.log({"training/mel_spec_error": mel_error, "steps": steps})

            early_stopping = False

            # Validation
            if steps % cfg.train["validation_interval"] == 0:  # and steps != 0:
                generator.eval()
                torch.cuda.empty_cache()
                val_err_tot = 0
                with torch.no_grad():
                    for j, batch in enumerate(validation_loader):


                        y_high, y_low, _ = batch

                        # y = y.unsqueeze(1)

                        y_g_hat = generator(y_low.to(device))
                        y_mel = torch.from_numpy(mel_spectrogram(y_high.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                                cfg.audio["hop"],
                                                cfg.audio["window_size"], cfg.audio["fmin"],
                                                cfg.audio["fmax_for_loss"]))
                        y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking = True))
                        y_g_hat_mel = torch.from_numpy(mel_spectrogram(y_g_hat.squeeze(1), cfg.audio["nfft"], cfg.audio["n_mels"],
                                                      cfg.audio["sr"], cfg.audio["hop"],
                                                      cfg.audio["window_size"], cfg.audio["fmin"],
                                                      cfg.audio["fmax_for_loss"]))
                        y_g_hat_mel = torch.autograd.Variable(y_g_hat_mel.to(device, non_blocking=True))


                        val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                        if j <= 4:
                            if steps == 0: #여기부터
                                pass

                    val_err = val_err_tot / (j + 1)
                    wandb.log({"validation/mel_spec_error": val_err, "steps": steps})


                    # Audio logging
                    y_high_np = y_high.cpu().detach().numpy().astype(float).reshape(-1)
                    y_low_np = y_low.cpu().detach().numpy().astype(float).reshape(-1)
                    y_g_hat_np = y_g_hat.cpu().detach().numpy().astype(float).reshape(-1)

                    gt_high_audio = wandb.Audio(y_high_np, sample_rate=cfg.audio["sr"], caption="ground_truth_high_audio")
                    pred_audio = wandb.Audio(y_g_hat_np, sample_rate=cfg.audio["sr"], caption="predicted_audio")
                    gt_low_audio = wandb.Audio(y_low_np, sample_rate=cfg.audio["sr"], caption="ground_truth_low_audio")
                    wandb.log({f"ground truth high audio in epoch {epoch}": gt_high_audio})
                    wandb.log({f"ground truth low audio in epoch {epoch}": gt_low_audio})
                    wandb.log({f"predicted audio in epoch {epoch}": pred_audio})

                    # Mel Spectrogram logging
                    gt_high_mel = librosa.power_to_db(mel_spectrogram(y_high.squeeze(), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                        cfg.audio["hop"],
                                        cfg.audio["window_size"], cfg.audio["fmin"],
                                        cfg.audio["fmax_for_loss"]), ref=np.max)
                    gt_low_mel = librosa.power_to_db(mel_spectrogram(y_low.squeeze(), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                        cfg.audio["hop"],
                                        cfg.audio["window_size"], cfg.audio["fmin"],
                                        cfg.audio["fmax_for_loss"]), ref=np.max)
                    pred_mel = librosa.power_to_db(mel_spectrogram(y_g_hat.squeeze(), cfg.audio["nfft"], cfg.audio["n_mels"], cfg.audio["sr"],
                                        cfg.audio["hop"],
                                        cfg.audio["window_size"], cfg.audio["fmin"],
                                        cfg.audio["fmax_for_loss"]), ref=np.max)
                    gt_high_mel = wandb.Image(
                        librosa.display.specshow(gt_high_mel, sr=cfg.audio["sr"], x_axis='time', y_axis='mel',
                                                 fmax=24000), caption = "gt_high_mel")
                    gt_low_mel = wandb.Image(librosa.display.specshow(gt_low_mel, sr=cfg.audio["sr"], x_axis='time',
                                                                      y_axis='mel', fmax=24000), caption = "gt_low_mel")
                    pred_mel = wandb.Image(librosa.display.specshow(pred_mel, sr=cfg.audio["sr"], x_axis='time',
                                                                    y_axis='mel', fmax=24000), caption = "predicted_mel")

                    wandb.log({f"ground truth high mel in epoch {epoch},": gt_high_mel})
                    wandb.log({f"ground truth low mel in epoch {epoch}": gt_low_mel})
                    wandb.log({f"predicted mel in epoch {epoch}": pred_mel})


                    if val_err < init_err:
                        early_stopping_count = 0
                        init_err = val_err
                    else:
                        early_stopping_count += 1

                    if early_stopping_count >= cfg.train["early_stopping_max"]:
                        print(f"Early stopping due to high validation error: \n {init_err} is the best validation error in the epoch: {epoch}, step : {steps}")
                        early_stopping = True
                        break
                    generator.train()

            # checkpointing
            if steps % cfg.train["checkpoint_interval"] == 0 and steps != 0:
                if best_err_for_ckpt > val_err:

                    best_err_for_ckpt = val_err
                    best_step = steps
                checkpoint_path = "{}/g_{:08d}".format(cfg.data["ckpt_save_path"], steps)
                save_checkpoint(checkpoint_path,
                                {'generator': generator.state_dict()})
                checkpoint_path = "{}/do_{:08d}".format(cfg.data["ckpt_save_path"], steps)
                save_checkpoint(checkpoint_path,
                                {'mpd': mpd.state_dict(),
                                 'msd': msd.state_dict(),
                                 'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                 'epoch': epoch})

                best_checkpoint_path = "{}/g_{:08d}".format(cfg.data["ckpt_save_path"], best_steps)


            if early_stopping:
                print(f"best checkpoint is : {best_checkpoint_path}")
                break



            steps += 1

        scheduler_g.step()
        scheduler_d.step()

        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:

        json_config = json.load(file)
        hyperparameters = AttrDict(json_config)

    print('Initializing Training Process..')

    torch.manual_seed(hyperparameters.info["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(hyperparameters.info["seed"])
        hyperparameters.train["n_gpu"] = torch.cuda.device_count()
        hyperparameters.train["batch_size"] = int(hyperparameters.train["batch_size"] / hyperparameters.train["n_gpu"])
        print('Batch size per GPU :',hyperparameters.train["batch_size"])
    else:
        pass



    train(0, hyperparameters)

if __name__ == "__main__":
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    main()