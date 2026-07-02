import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from mlp import MLP


class Patchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, c_in: int, dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.net = nn.Sequential(
            # Initial convolution
            nn.Conv2d(c_in, dim, kernel_size=patch_size, stride=patch_size),    # (b, c, s, s) --> (b, d, s/p, s/p)

            # Patchify
            Rearrange("b d h w -> b (h w) d"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Depatchifier(nn.Module):
    def __init__(self, img_size: int, patch_size: int, dim: int,
                 final_dim: int, c_out: int):
        super().__init__()
        self.patch_size = patch_size
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        h = w = img_size // patch_size


        self.net = nn.Sequential(
            # Norm + MLP
            nn.RMSNorm(dim, elementwise_affine=False),
            MLP([dim, 4*dim, final_dim * patch_size ** 2]), # (b, 64, 160)

            # Depatchify
            Rearrange("b (h w) (f ph pw) -> b f (h ph) (w pw)", h=h, w=w,
                      f=final_dim, ph=patch_size, pw=patch_size),   # (b, 64, 160) --> (b, 10, 32, 32)

            # Final convolution
            nn.Conv2d(final_dim, c_out, kernel_size=3, padding=1)   # (b, 10, 32, 32) --> (b, 1, 32, 32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
