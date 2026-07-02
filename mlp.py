import torch
import torch.nn as nn
from typing import List, Type


class MLP(nn.Module):
    def __init__(
        self, 
        dims: List[int], 
        activation: Type[torch.nn.Module] = torch.nn.SiLU, 
        final_init: bool = False
    ):
        super().__init__()
        mlp = []
        
        for idx in range(len(dims) - 1):
            mlp.append(nn.Linear(dims[idx], dims[idx + 1]))
            if idx < len(dims) - 2:
                mlp.append(activation())
        
        self.net = torch.nn.Sequential(*mlp)

        if final_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        - x: b n d
        Returns:
        - x: b n d
        """
        return self.net(x)
