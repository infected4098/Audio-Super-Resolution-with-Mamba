import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from zeta.nn import SSM
from einops.layers.torch import Reduce
# Modified from https://github.com/kyegomez/VisionMamba/blob/main/vision_mamba/model.py
from tqdm import tqdm

class VisionMambaBlock(nn.Module):
    """
    Args:
        dim (int): The input dimension of the input tensor.
        dt_rank (int): The rank of the state space model.
        dim_inner (int): The dimension of the inner layer of the
            multi-head attention.
        d_state (int): The dimension of the state space model.
    """

    def __init__(
        self,
        dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = dim # 256
        self.dt_rank = dt_rank # defaults to 32
        self.dim_inner = dim_inner # 512
        self.d_state = d_state # 256

        self.forward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.backward_conv1d = nn.Conv1d(
            in_channels=dim, out_channels=dim, kernel_size=1
        )
        self.norm = nn.LayerNorm(dim)
        # Linear layer for z and x
        self.proj_x = nn.Linear(dim, dim)
        self.proj_z = nn.Linear(dim, dim)

        self.silu = nn.SiLU()
        self.ssm_forward = SSM(dim, dt_rank, dim_inner, d_state)
        self.ssm_backward = SSM(dim, dt_rank, dim_inner, d_state)


        # Softplus
        self.softplus = nn.Softplus()

    def forward(self, x: torch.Tensor):
        b, s, d = x.shape # [batch_size, sequence_length, hidden_dim]
        # Skip connection
        skip = x

        # Normalization
        x = self.norm(x)

        # Split x into x1 and x2 with linears
        z1 = self.proj_z(x)
        x = self.proj_x(x)


        # Forward
        x1 = x
        x1 = rearrange(x1, "b s d -> b d s")
        x1 = self.softplus(self.forward_conv1d(x1))
        x1 = rearrange(x1, "b d s -> b s d")
        x1 = self.ssm_forward(x1)
        # Backward
        x2 = torch.flip(x, [1])
        x2 = rearrange(x2, "b s d -> b d s")
        x2 = self.softplus(self.backward_conv1d(x2))
        x2 = rearrange(x2, "b d s -> b s d")
        x2 = self.ssm_backward(x2)
        x2 = torch.flip(x2, [1])


        # Activation
        z = self.silu(z1)

        # Matmul
        x1 *= z
        x2 *= z

        # Residual connection
        return x1 + x2 + skip

# model = VisionMambaBlock(dim = 384, dt_rank = 32, dim_inner = 384, d_state = 384).to("cuda:0")

