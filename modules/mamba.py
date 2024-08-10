import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from modules.embedding import MelFreeStemEmbed1D
import hparams
from modules.resblock import ResBlock1
from mamba_ssm import Mamba
import einops
from utils import init_weights, get_padding
LRELU_SLOPE = 0.1

class MambaEncoder(nn.Module):
    def __init__(self, cfg):
        super(MambaEncoder, self).__init__()
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