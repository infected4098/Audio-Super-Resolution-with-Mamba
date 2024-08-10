import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from modules.embedding import MelFreeStemEmbed1D, MelFreeStemEmbed1D_res, MelFreeStemEmbedMamba
import hparams
from modules.resblock import ResBlock1
from mamba_ssm import Mamba
import einops
from utils import init_weights, get_padding, summarize_model
LRELU_SLOPE = 0.1
from modules.mamba import MambaEncoder

class asmr_full(nn.Module):
    def __init__(self, cfg_embed, cfg_hifigan, cfg_mamba): # (B, T, C) #cfg_embed = hparams.embed, cfg_hifigan = hparams.hifigan
        super(asmr_full, self).__init__()

        # Feature Extraction
        self.num_kernels = len(cfg_hifigan["resblock_kernel_sizes"])  # 3, 7, 11 --> 3
        self.num_upsamples = len(cfg_hifigan["upsample_rates"])
        self.emb = MelFreeStemEmbedMamba(in_channels = 1, kernel_size = cfg_embed["cnn_stem_kernel_sizes_1D"],
                    embed_dim = cfg_embed["cnn_stem_channels_1D"], strides = cfg_embed["cnn_stem_strides_1D"], cfg_mamba = cfg_mamba)
        # Upsampling
        resblock = ResBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg_hifigan["upsample_rates"], cfg_hifigan["upsample_kernel_sizes"])):
            # [[4, 8], [4, 8], [2, 4], [2, 4], [2, 4]]
            self.ups.append(weight_norm(
                ConvTranspose1d(cfg_hifigan['upsample_initial_channel'] // (2 ** i),
                                cfg_hifigan["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))  # upsampling ratio = u, exact upsampling by u with the certain padding

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg_hifigan["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(cfg_hifigan["resblock_kernel_sizes"],
                                           cfg_hifigan["resblock_dilation_sizes"])):  # [[3, [1, 3, 5], [7, [1, 3, 5]. ...]]]
                self.resblocks.append(resblock(cfg_hifigan, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    def forward(self, x):
        x, res_lst = self.emb(x) # [B, 1, sr*T] --> [B, embed_dim[-1], sr*T/prod(strides)]
        res_lst = res_lst[::-1]
        for i in range(self.num_upsamples): #여기부터
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            x = x + res_lst[i]
        x = F.leaky_relu(x)
        x = self.conv_post(x)  # [B, cfg_hifigan.upsample_initial_channel // (2 ** (i + 1)), sr*T] --> [B, 1, 256*T]
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_post)
"""import json
from env import AttrDict
with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:

    json_config = json.load(file)
    hyperparameters = AttrDict(json_config)
    cfg = hyperparameters
    generator = asmr_full(cfg.embed, cfg.hifigan, cfg.mamba).to("cuda:0")
    x = torch.rand([16, 1, 128*400]).to("cuda:0")
    y = generator(x)
    print(y.shape)"""

class asmr_res(nn.Module):
    def __init__(self, cfg_embed, cfg_hifigan): # (B, T, C) #cfg_embed = hparams.embed, cfg_hifigan = hparams.hifigan
        super(asmr_res, self).__init__()

        # Feature Extraction
        self.num_kernels = len(cfg_hifigan["resblock_kernel_sizes"])  # 3, 7, 11 --> 3
        self.num_upsamples = len(cfg_hifigan["upsample_rates"])
        self.emb = MelFreeStemEmbed1D_res(in_channels = 1, kernel_size = cfg_embed["cnn_stem_kernel_sizes_1D"],
                    embed_dim = cfg_embed["cnn_stem_channels_1D"], strides = cfg_embed["cnn_stem_strides_1D"])
        #self.mambaencoder = MambaEncoder(hparams.mamba)

        #
        resblock = ResBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg_hifigan["upsample_rates"], cfg_hifigan["upsample_kernel_sizes"])):
            # [[4, 8], [4, 8], [2, 4], [2, 4], [2, 4]]
            self.ups.append(weight_norm(
                ConvTranspose1d(cfg_hifigan['upsample_initial_channel'] // (2 ** i),
                                cfg_hifigan["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))  # upsampling ratio = u, exact upsampling by u with the certain padding

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg_hifigan["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(cfg_hifigan["resblock_kernel_sizes"],
                                           cfg_hifigan["resblock_dilation_sizes"])):  # [[3, [1, 3, 5], [7, [1, 3, 5]. ...]]]
                self.resblocks.append(resblock(cfg_hifigan, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    def forward(self, x):
        x, res_lst = self.emb(x) # [B, 1, sr*T] --> [B, embed_dim[-1], sr*T/prod(strides)]
        res_lst = res_lst[::-1]
        #x = einops.rearrange(x, 'b d l -> b l d') # [B, sr*T/prod(strides), embed_dim[-1]]
        #x = self.mambaencoder(x) # [B, sr*T/prod(strides), embed_dim[-1]] --> [B, sr*T/prod(strides), embed_dim[-1]]
        #x = einops.rearrange(x, 'b l d -> b d l') # [B, embed_dim[-1], sr*T/prod(strides)]
        for i in range(self.num_upsamples): #여기부터
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
            x = x + res_lst[i]
        x = F.leaky_relu(x)
        x = self.conv_post(x)  # [B, cfg_hifigan.upsample_initial_channel // (2 ** (i + 1)), sr*T] --> [B, 1, 256*T]
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        #remove_weight_norm(self.emb)
        remove_weight_norm(self.conv_post)


class asmr(nn.Module):
    def __init__(self, cfg_embed, cfg_hifigan): # (B, T, C) #cfg_embed = hparams.embed, cfg_hifigan = hparams.hifigan
        super(asmr, self).__init__()

        # Feature Extraction
        self.num_kernels = len(cfg_hifigan["resblock_kernel_sizes"])  # 3, 7, 11 --> 3
        self.num_upsamples = len(cfg_hifigan["upsample_rates"])
        self.emb = MelFreeStemEmbed1D(in_channels = 1, kernel_size = cfg_embed["cnn_stem_kernel_sizes_1D"],
                    embed_dim = cfg_embed["cnn_stem_channels_1D"], strides = cfg_embed["cnn_stem_strides_1D"])
        #self.mambaencoder = MambaEncoder(hparams.mamba)

        #
        resblock = ResBlock1
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(cfg_hifigan["upsample_rates"], cfg_hifigan["upsample_kernel_sizes"])):
            # [[4, 8], [4, 8], [2, 4], [2, 4], [2, 4]]
            self.ups.append(weight_norm(
                ConvTranspose1d(cfg_hifigan['upsample_initial_channel'] // (2 ** i),
                                cfg_hifigan["upsample_initial_channel"] // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))  # upsampling ratio = u, exact upsampling by u with the certain padding

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = cfg_hifigan["upsample_initial_channel"] // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(cfg_hifigan["resblock_kernel_sizes"],
                                           cfg_hifigan["resblock_dilation_sizes"])):  # [[3, [1, 3, 5], [7, [1, 3, 5]. ...]]]
                self.resblocks.append(resblock(cfg_hifigan, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)
    def forward(self, x):
        x = self.emb(x) # [B, 1, sr*T] --> [B, embed_dim[-1], sr*T/prod(strides)]
        #x = einops.rearrange(x, 'b d l -> b l d') # [B, sr*T/prod(strides), embed_dim[-1]]
        #x = self.mambaencoder(x) # [B, sr*T/prod(strides), embed_dim[-1]] --> [B, sr*T/prod(strides), embed_dim[-1]]
        #x = einops.rearrange(x, 'b l d -> b d l') # [B, embed_dim[-1], sr*T/prod(strides)]
        for i in range(self.num_upsamples): #여기부터
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)  # [B, 512, sr*T] --> [B, 32, 256*T]
        x = self.conv_post(x)  # [B, cfg_hifigan.upsample_initial_channel // (2 ** (i + 1)), sr*T] --> [B, 1, 256*T]
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        #remove_weight_norm(self.emb)
        remove_weight_norm(self.conv_post)


"""
import json
from env import AttrDict
with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:

    json_config = json.load(file)
    hyperparameters = AttrDict(json_config)
    cfg = hyperparameters
    generator = asmr_res(cfg.embed, cfg.hifigan)
    x = torch.rand([16, 1, 128*400])
    y = generator(x)
    print(y.shape)
    """


class DiscriminatorP(torch.nn.Module):
    """input_shape = [B, 1, sequence_length]"""
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period) # [B, C, H, W]

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap # [len(self.convs), C_l, H, W]


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
