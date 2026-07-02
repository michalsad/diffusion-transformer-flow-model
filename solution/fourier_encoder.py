import math
import torch
import torch.nn as nn


class FourierEncoder(nn.Module):
    """
    Based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/karras_unet.py#L183
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(1, self.half_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - t: b
        Returns:
        - embeddings: b d
        """
        # Step 1: compute frequencies f_i = 2 * pi * w_i * t
        t = t.view(-1, 1) # b 1
        freqs = t * self.weights * 2 * math.pi # b hd

        # Step 2: compute sin(f_i) and cos(f_i)
        sin_embed = torch.sin(freqs) # b hd
        cos_embed = torch.cos(freqs) # b hd

        # Step 3: Concatenate and return
        return torch.cat([sin_embed, cos_embed], dim=-1) * math.sqrt(2) # b d