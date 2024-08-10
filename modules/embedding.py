import torch
from einops import rearrange
from torch import nn
from modules.normalization import LayerNorm
import torch.nn.functional as F
from mamba_ssm import Mamba
import numbers
import hparams
class MambaBlock(nn.Module):
    def __init__(self, cfg):
        super(MambaBlock, self).__init__()
        self.d_model = cfg["d_model"]
        self.d_state = cfg["d_state"]
        self.d_conv = cfg["d_conv"]
        self.expand = cfg["expand"]
        self.n_blocks = cfg["n_blocks"]
        self.MambaBlocks = nn.ModuleList()
        for _ in range(self.n_blocks):
            self.MambaBlocks.append(Mamba(d_model = self.d_model, d_state = self.d_state,
                                          d_conv = self.d_conv, expand = self.expand))
    def forward(self, x):
        for i in range(self.n_blocks):
            resid = x
            x = self.MambaBlocks[i](x)
            x = x + resid
        return x

class ConvMambaBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, cfg_mamba):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.d_state = cfg_mamba["d_state"]
        self.d_conv = cfg_mamba["d_conv"]
        self.expand = cfg_mamba["expand"]
        self.conv = nn.Conv1d(self.in_channels, self.out_channels,
                                       kernel_size = self.kernel_size, stride = self.strides, padding = (self.kernel_size-self.strides)//2)
        self.convln = LayerNorm(self.out_channels, (0, 2, 1))
        self.mamba1 = Mamba(d_model = self.out_channels, d_state = self.d_state,
                                          d_conv = self.d_conv, expand = self.expand)
        self.ln1 = LayerNorm(self.out_channels, (0, 2, 1))
        self.mamba2 = Mamba(d_model = self.out_channels, d_state = self.d_state,
                                          d_conv = self.d_conv, expand = self.expand)
        self.ln2 = LayerNorm(self.out_channels, (0, 2, 1))

    def forward(self, x):
        x = self.conv(x)
        res1 = x
        x = self.convln(x)
        x = x + res1
        x = rearrange(x, 'b d l -> b l d')
        x = self.mamba1(x)
        x = rearrange(x, 'b l d -> b d l')
        res2 = x
        x = self.ln1(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.mamba2(x)
        x = rearrange(x, 'b l d -> b d l')
        x = res2 + x
        x = self.ln2(x)
        return x

        #x = einops.rearrange(x, 'b d l -> b l d') # [B, sr*T/prod(strides), embed_dim[-1]]
        #x = self.mambaencoder(x) # [B, sr*T/prod(strides), embed_dim[-1]] --> [B, sr*T/prod(strides), embed_dim[-1]]
        #x = einops.rearrange(x, 'b l d -> b d l') # [B, embed_dim[-1], sr*T/prod(strides)]


class MelFreeConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.conv = nn.Conv1d(self.in_channels, self.out_channels,
                                       kernel_size = self.kernel_size, stride = self.strides, padding = (self.kernel_size-self.strides)//2)
        self.ln = LayerNorm(self.out_channels, (0, 2, 1))
    def forward(self, x):
        x = self.conv(x)
        x = self.ln(x)
        return x

class MelFreeStemEmbedMamba(nn.Module): #TBD
    def __init__(self, in_channels,
                 kernel_size, embed_dim, strides, cfg_mamba):
        """
        audio_size = T * sr; temporal_frame_num
        in_channels = 1
        kernel_size = [20, 16, 8, 8, 8]
        embed_dim = [24, 48, 96, 192, 384]
        strides = [4, 4, 2, 2, 2]
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.strides = strides
        self.cnn_stem = nn.ModuleList()
        self.cnn_stem.append(MelFreeConvBlock(in_channels, embed_dim[0], kernel_size[0], strides[0]))
        self.cfg_mamba = cfg_mamba
        for i in range(0, int(len(kernel_size)) - 2):
            self.cnn_stem.append(ConvMambaBlock(self.embed_dim[i], self.embed_dim[i+1],
                                       kernel_size = self.kernel_size[i+1], strides = self.strides[i+1], cfg_mamba = self.cfg_mamba))

        self.cnn_stem.append(nn.Conv1d(self.embed_dim[-2], self.embed_dim[-1], kernel_size = self.kernel_size[-1],
                                       stride=self.strides[-1], padding = (self.kernel_size[-1]-self.strides[-1])//2))
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x): # [batch_size, 1, audio_size]
        res_lst = []
        res_lst.append(x)
        for i, block in enumerate(self.cnn_stem):
            x = block(x)
            res_lst.append(x)
            if i < len(self.cnn_stem) - 1:

                x = self.leakyrelu(x)

        return x, res_lst[:-1] # no skip connection for the shortest cut.

"""import json
from env import AttrDict
with open("/home/prml/.virtualenvs/mamba_super_1/lib/python3.11/configs/cfgs.json", "r") as file:

    json_config = json.load(file)
    hyperparameters = AttrDict(json_config)
    cfg = hyperparameters
    generator = MelFreeStemEmbedMamba(in_channels = 1, kernel_size = cfg.embed["cnn_stem_kernel_sizes_1D"],
                    embed_dim = cfg.embed["cnn_stem_channels_1D"], strides = cfg.embed["cnn_stem_strides_1D"], cfg_mamba = cfg.mamba).to("cuda:0")
    x = torch.rand([16, 1, 128*400]).to("cuda:0")
    y = generator(x)
    print(y[0].shape)"""




class MelFreeStemEmbed1D_res(nn.Module): #TBD
    def __init__(self, in_channels,
                 kernel_size, embed_dim, strides):
        """
        audio_size = T * sr; temporal_frame_num
        in_channels = 1
        kernel_size = [20, 16, 8, 8, 8]
        embed_dim = [24, 48, 96, 192, 384]
        strides = [4, 4, 2, 2, 2]
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.strides = strides
        self.cnn_stem = nn.ModuleList()
        self.cnn_stem.append(MelFreeConvBlock(in_channels, embed_dim[0], kernel_size[0], strides[0]))
        for i in range(0, int(len(kernel_size)) - 2):
            self.cnn_stem.append(MelFreeConvBlock(self.embed_dim[i], self.embed_dim[i+1],
                                       kernel_size = self.kernel_size[i+1], strides = self.strides[i+1]))

        self.cnn_stem.append(nn.Conv1d(self.embed_dim[-2], self.embed_dim[-1], kernel_size = self.kernel_size[-1],
                                       stride=self.strides[-1], padding = (self.kernel_size[-1]-self.strides[-1])//2))
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x): # [batch_size, 1, audio_size]
        res_lst = []
        res_lst.append(x)
        for i, block in enumerate(self.cnn_stem):
            x = block(x)
            res_lst.append(x)
            if i < len(self.cnn_stem) - 1:

                x = self.leakyrelu(x)

        return x, res_lst[:-1] # no skip connection for the shortest cut.


"""emb = MelFreeStemEmbed1D_res(in_channels = 1, kernel_size = hparams.embed["cnn_stem_kernel_sizes_1D"],
                embed_dim = hparams.embed["cnn_stem_channels_1D"], strides = hparams.embed["cnn_stem_strides_1D"])
a = torch.rand([32, 1, 128*400])
b, res_lst = emb(a)
print(b.shape, res_lst[0].shape, res_lst[1].shape, res_lst[2].shape, res_lst[3].shape)"""
class MelFreeStemEmbed1D(nn.Module):
    def __init__(self, in_channels,
                 kernel_size, embed_dim, strides):
        """
        audio_size = T * sr; temporal_frame_num
        in_channels = 1
        kernel_size = [24, 48, 96, 192, 384]
        embed_dim = [32, 64, 128, 256, 512]
        strides = [4, 3, 3, 2, 1, 1]
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.embed_dim = embed_dim
        self.strides = strides
        self.cnn_stem = nn.Sequential()
        self.cnn_stem.append(nn.Conv1d(self.in_channels, self.embed_dim[0],
                                       kernel_size = self.kernel_size[0], stride = self.strides[0], padding = (self.kernel_size[0]-self.strides[0])//2))
        self.cnn_stem.append(LayerNorm(self.embed_dim[0], (0, 2, 1)))
        self.cnn_stem.append(nn.LeakyReLU())
        for i in range(0, int(len(kernel_size)) -2):

            self.cnn_stem.append(nn.Conv1d(self.embed_dim[i], self.embed_dim[i+1],
                                           kernel_size = self.kernel_size[i+1], stride = self.strides[i+1], padding = (self.kernel_size[i+1]-self.strides[i+1])//2))
            self.cnn_stem.append(LayerNorm(self.embed_dim[i+1], (0, 2, 1)))
            self.cnn_stem.append(nn.LeakyReLU())
        self.cnn_stem.append(nn.Conv1d(self.embed_dim[-2], self.embed_dim[-1], kernel_size = self.kernel_size[-1],
                                       stride=self.strides[-1], padding = (self.kernel_size[-1]-self.strides[-1])//2))

    def forward(self, x): # [batch_size, 1, audio_size]
        x = self.cnn_stem(x)
        return x # [batch_size, embed_dim[-1], about 600 frames per seconds]





