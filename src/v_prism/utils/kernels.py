from typing import Protocol

import torch
from torch import Tensor


class Kernel(Protocol):
    """A Protocol class for a kernel that maps (N, D) x (M, D) -> (N, M)"""
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x: (N, D)
            y: (M, D)

        Returns:
            k: (N, M) kernel matrix    
        """
        raise NotImplementedError()
    


class GaussianKernel(Kernel):
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.exp(- self.gamma * _square_dists(x, y))
        

class LaplacianKernel(Kernel):
    def __init__(self, gamma: float) -> None:
        self.gamma = gamma

    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.exp(- self.gamma * torch.sqrt(_square_dists(x, y)))

def _square_dists(x: Tensor, y: Tensor) -> Tensor:
    square_dists =  torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(y ** 2, dim=-1).unsqueeze(-2) - 2 * (x @ y.transpose(-1, -2))
    return torch.maximum(square_dists, torch.zeros(1, dtype=x.dtype, device=x.device))  # for numerical stability w sqrt

