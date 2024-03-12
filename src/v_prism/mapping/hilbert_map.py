import torch
from torch import Tensor
from torch.nn import Linear, CrossEntropyLoss
from torch.optim import Optimizer, Adam, SGD

from ..utils.kernels import Kernel


def _get_optimizer(optim_type: str, params: object, lr: float) -> Optimizer:
    if optim_type == 'adam':
        return Adam(params, lr=lr)
    elif optim_type == 'sgd':
        return SGD(params, lr=lr)
    else:
        raise NotImplementedError()

class HilbertMap:
    def __init__(
        self,
        hinge_points: Tensor,
        kernel: Kernel,
        *,
        optimizer_type: str = 'adam',
        lr: float = 0.01,
        num_iterations: int = 100,
    ) -> None:
        H, D = hinge_points.shape
        self.device = hinge_points.device
        self.points = torch.zeros((0, D), device=self.device, dtype=hinge_points.dtype)
        self.labels = torch.zeros((0,), device=self.device, dtype=hinge_points.dtype)
        raise NotImplementedError()


    def update(self, X: Tensor, y: Tensor) -> None:
        """updates map with new data
        
        Args:
            X: (P, 3) float array of x values for new points
            y: array of y values for new points
        """
        raise NotImplementedError()
    
    def predict(self, X: Tensor) -> Tensor:
        """predicts y values for given X array"""
        raise NotImplementedError()


class HilbertMapSoftmax:
    def __init__(
        self,
        hinge_points: Tensor,
        kernel: Kernel,
        num_classes: int,
        *,
        optimizer_type: str = 'adam',
        lr: float = 0.01,
        num_iterations: int = 100,
        batch_size: int = 128,
        bias: bool = True
    ) -> None:
        H, D = hinge_points.shape
        self.hinge_points = hinge_points
        self.device = hinge_points.device
        self.points = torch.zeros((0, D), device=self.device, dtype=hinge_points.dtype)
        self.labels = torch.zeros((0,), device=self.device, dtype=torch.int64)
        self.kernel = kernel
        self.W = Linear(H, num_classes, device=self.device, dtype=hinge_points.dtype, bias=bias)
        self.optimizer = _get_optimizer(optimizer_type, self.W.parameters(), lr=lr)
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.loss = CrossEntropyLoss()
        self.num_classes = num_classes


    def _get_feats(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        feats = self.kernel(X, self.hinge_points)
        return feats

    def update(self, X: Tensor, y: Tensor, *, print_every_t: int = 50) -> None:
        """updates map with new data
        
        Args:
            X: (P, 3) float array of x values for new points
            y: array of y values for new points
        """
        self.W.train()
        self.points = torch.concat([self.points, X], dim=0)
        self.labels = torch.concat([self.labels, y], dim=0).to(torch.int64)

        for i in range(self.num_iterations):
            self.optimizer.zero_grad()
            idx = torch.randint(0, self.points.shape[0], size=(self.batch_size,))
            feats = self._get_feats(self.points[idx])
            labels = self.labels[idx]
            scores = self.W(feats)
            loss: Tensor = self.loss(scores, labels)
            loss.backward()
            self.optimizer.step()
            if (i + 1) % print_every_t == 0:
                print(f'iteration {i + 1} has loss: {loss.detach().cpu().item()}')


    
    def predict(self, X: Tensor) -> Tensor:
        """predicts y values for given X array"""
        self.W.eval()
        with torch.no_grad():
            return torch.softmax(self.W(self._get_feats(X)), dim=1)




