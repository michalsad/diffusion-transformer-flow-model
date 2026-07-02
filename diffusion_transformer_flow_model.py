import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from mlp import MLP
from fourier_encoder import FourierEncoder
from patchifier import Patchifier, Depatchifier
    

class Head(nn.Module):
    """
    Attention(k, q, v) = softmax(q @ k.T / sqrt(d_h)) @ v
    """
    def __init__(self, dim, head_size):
        super().__init__()

        # Initialize keys, queries, and values


        self.scale = head_size**-0.5

    def forward(self, x):
        # Compute keys, queries, and values


        # Compute attention scores ("affinities")


        # Weighted aggregation of the values

        
        pass


class MHA(nn.Module):
    def __init__(self, dim, n_heads, head_size):
        super().__init__()
        assert dim % n_heads == 0

        self.heads = nn.ModuleList(
            [Head(dim, head_size) for _ in range(n_heads)]
        )

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # Project to input dimensions
        out = self.proj(out)
        return out


def modulate(x: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + bias


class DiffusionTransformerLayer(nn.Module):
    def __init__(self, dim: int, heads: int):
        """
        Args:
        - n_tokens: sequence length (for sake of positional embeddings)
        - dim: dimension of hidden layers
        - heads: number of attention heads
        """
        super().__init__()

        # Normalization
        # Root Mean Square Normalization: RMSNorm(x) = x / sqrt(mean(x^2) + eps)
        # Cheaper to compute than LayerNorm
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        # Adaptive layer norm (AdaLN) 
        self.ada_ln = nn.Sequential(
            nn.RMSNorm(dim, elementwise_affine=False),
            nn.Linear(dim, dim * 6)
        )

        # Initialize conditioning to zero - stabilizes residual connection!
        nn.init.zeros_(self.ada_ln[1].weight)
        nn.init.zeros_(self.ada_ln[1].bias)

        # Attention
        self.attn = MHA(dim, n_heads=heads, head_size=dim//heads)

        # Feedforward
        self.ff = MLP([dim, 4 * dim, dim])

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Compute conditioning gating, scaling, and bias
        # use label and time embeddings to produce per-channel scale/shift 
        # parameters that modulate normalized activations
        c = rearrange(self.ada_ln(c), 'b d -> b 1 d') # b 1 d
        attn_scale, attn_bias, attn_gate, ff_scale, ff_bias, ff_gate = c.chunk(6, dim=-1)

        # Attention
        x = x + attn_gate * self.attn(
            modulate(self.norm1(x), attn_scale, attn_bias)
            )
        
        # Feedforward
        x = x + ff_gate * self.ff(
            modulate(self.norm2(x), ff_scale, ff_bias)
            )

        return x
    

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        n_tokens: int,
        dim: int,
        **layer_kwargs,
    ):
        """
        Args:
        - n_tokens: sequence length (for sake of positional embeddings)
        - dim: dimension of hidden layers
        - heads: number of attention heads
        - depth: number of layers
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.layers.append(
                DiffusionTransformerLayer(dim=dim, **layer_kwargs)
            )

        # Positional encodings
        self.pos_encodings = nn.Parameter(torch.randn(n_tokens, dim))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encodings.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, c)
        return x
    

class DiffusionTransformerFlowModel(nn.Module):
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        num_layers: int = 12,
        c: int = 1,
        dim: int = 256,
        heads: int = 4,
        final_dim: int = 10,
        n_classes: int = 11,
    ):
        super().__init__()
        # 0. Construct time_embedder and y_embedder
        self.time_embedder = FourierEncoder(dim)
        self.y_embedder = nn.Embedding(num_embeddings = n_classes, 
                                       embedding_dim = dim)

        # 1. Construct patchifier
        self.patchifier = Patchifier(
            img_size=img_size,  # s
            patch_size=patch_size,  # p
            c_in=c, # c
            dim=dim # d
        )   # (b, c, s, s) --> (b, s/p * s/p, d)

        # 2. Construct DiT
        n_tokens = (img_size // patch_size) ** 2
        self.dit = DiffusionTransformer(
            depth=num_layers,
            n_tokens=n_tokens,
            dim=dim,
            heads=heads,
        )   # (b, s/p * s/p, d) --> (b, s/p * s/p, d)

        # 3. Construct de-patchifier
        self.depatchifier = Depatchifier(
            img_size=img_size,
            patch_size=patch_size,
            dim=dim,
            final_dim=final_dim,
            c_out=c
        )   # (b, s/p * s/p, d) --> (b, c, s, s)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b 1 32 32
        - t: b 1 1 1
        - c: b 1 1 1
        Returns:
        - u_t^theta(x|y): b 1 32 32
        """
        # 1. Embed time and y
        t_embed = self.time_embedder(t) # b d
        y_embed = self.y_embedder(y) # b d

        # 2. Patchify
        x = self.patchifier(x) # b n d

        # 3. Pass through DiT
        x = self.dit(x, t_embed + y_embed) # b d

        # 4. Depatchify
        x = self.depatchifier(x) # b 1 32 32

        return x