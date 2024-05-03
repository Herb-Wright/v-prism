
import torch
from torch import Tensor
import numpy as np

from ..utils.kernels import Kernel


def _lambda(xi: Tensor) -> Tensor:
    return (0.5 - torch.sigmoid(xi)) / (2*xi)

class FastBayesianHilbertMap:
    """Sequential Bayesian Hilbert Map - Diagonal Covariance"""
    def __init__(
        self, 
        hinge_points: Tensor,
        kernel: Kernel,
        *, 
        num_iterations: int = 1,
        num_iterations_initial: int | None = None,
        sigma_0: float | Tensor = 10000.0, 
        mu_0: float | Tensor = 0.0,
    ) -> None:
        """
        Args:
            - hinge_points: (H, 3) array of hinge points.
            - num_iterations: (int) the number of iterations to perform EM algorithm each update. 
            defaults to 1.
            - (optional) num_iterations_initial: num of iterations on first update. If `None`, then
            it will be set to num_iterations.
            - (optional) sigma_0: (float) or (H, H) the initial variance matrix/value. defaults to
            1e4.
            - (optional) mu_0: (float) or (H,) the initial mean. defaults to 0
        """
        H = hinge_points.shape[0]
        self.device = hinge_points.device
        if isinstance(sigma_0, float) or isinstance(sigma_0, int):
            sigma_0 = torch.ones(H + 1, device=self.device, dtype=hinge_points.dtype) * sigma_0
        if isinstance(mu_0, float) or isinstance(mu_0, int):
            mu_0 = torch.ones(H + 1, device=self.device, dtype=hinge_points.dtype) * mu_0
        self.sigma = sigma_0
        self.mu = mu_0
        self.hinge_points = hinge_points
        self.num_iterations = num_iterations
        self.num_iterations_initial = num_iterations_initial
        self.kernel = kernel

    def _get_feats(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        feats = self.kernel(X, self.hinge_points)
        return torch.concatenate([feats, torch.ones_like(X[:, 0:1])], dim=1)
    

    def update(self, X: Tensor, y: Tensor) -> None:
        """updates the BHM

        Performs the E-M optimization algorithm for the given (X, y) data. Uses the variational
        parameter xi to lower bound the sigmoid likelihood with a gaussian.

        Args:
            - X: (P, 3)
            - y: (P,) binary array corresponding to occupied (1) and empty (0)
        """
        P = X.shape[0]
        H = self.hinge_points.shape[0] + 1
        y = y.to(self.hinge_points.dtype)
        num_iterations = (
            self.num_iterations
        ) if self.num_iterations_initial is None else self.num_iterations_initial
        self.num_iterations_initial = None
        sigma_inv_0 = 1 / self.sigma
        mu_0 = self.mu
        feats = self._get_feats(X)  # (P, H+1)

        # E-M algo
        xi = torch.ones(P, device=self.device, dtype=X.dtype)  # should this be 0?
        for i in range(num_iterations):
            # E step
            sigma_inv = sigma_inv_0 + 2 * torch.sum(feats.T * torch.abs(_lambda(xi)) * feats.T, dim=1)
            sigma = 1 / sigma_inv

            mu = sigma * (sigma_inv_0 * mu_0 + feats.T @ (y - 0.5))

            # M step
            xi = torch.sqrt(torch.sum((feats ** 2) * sigma, dim=1) + (feats @ mu) ** 2)
            
        self.sigma = sigma  # (H+1)
        self.mu = mu  # (H+1,)


    def predict(self, X: Tensor) -> Tensor:
        """returns expected occupancy probability

        Approximates E_w [sigmoid(w^T \phi(x))].
        
        Args:
            - X: (Q, 3) query points
        
        Returns:
            - expected_y: (Q,) expected occupany probability corresponding to query points
        """
        feats = self._get_feats(X)  # (Q, H+1)
        Q, H = feats.shape
        means =  feats @ self.mu # (Q,)

        variances = torch.sum((feats * self.sigma) * feats, dim=1)  # (Q,)

        return torch.sigmoid(means / torch.sqrt(1 + (torch.pi / 8) * variances))  # (Q,)
    
    def to(self, device: torch.device):
        return FastBayesianHilbertMap(
            self.hinge_points.to(device),
            kernel=self.kernel,
            num_iterations=self.num_iterations,
            num_iterations_initial=self.num_iterations_initial,
            sigma_0=self.sigma.to(device),
            mu_0=self.mu.to(device),
        )


class BayesianHilbertMapWithFullCovarianceMatrixNoInv:
    """Sequential Bayesian Hilbert Map - Full Covariance"""
    def __init__(
        self, 
        hinge_points: Tensor,
        kernel: Kernel,
        *, 
        num_iterations: int = 1,
        num_iterations_initial: int | None = None,
        sigma_0: float | Tensor = 10000.0, 
        mu_0: float | Tensor = 0.0,
        sigma_is_inv: bool = False
    ) -> None:
        """
        Args:
            - hinge_points: (H, 3) array of hinge points.
            - num_iterations: (int) the number of iterations to perform EM algorithm each update. 
            defaults to 1.
            - (optional) num_iterations_initial: num of iterations on first update. If `None`, then
            it will be set to num_iterations.
            - (optional) sigma_0: (float) or (H, H) the initial variance matrix/value. defaults to
            1e4.
            - (optional) mu_0: (float) or (H,) the initial mean. defaults to 0
        """
        H = hinge_points.shape[0]
        self.device = hinge_points.device
        if isinstance(sigma_0, float) or isinstance(sigma_0, int):
            sigma_0_inv = torch.eye(H + 1, device=self.device, dtype=hinge_points.dtype) / sigma_0
        if isinstance(mu_0, float) or isinstance(mu_0, int):
            mu_0 = torch.ones(H + 1, device=self.device, dtype=hinge_points.dtype) * mu_0
        if sigma_is_inv:
            self.sigma_inv = sigma_0
        else:
            self.sigma_inv = sigma_0_inv
        self.mu = mu_0
        self.hinge_points = hinge_points
        self.num_iterations = num_iterations
        self.num_iterations_initial = num_iterations_initial
        self.kernel = kernel

    def _get_feats(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        feats = self.kernel(X, self.hinge_points)
        return torch.concatenate([feats, torch.ones_like(X[:, 0:1])], dim=1)

    def update(self, X: Tensor, y: Tensor) -> None:
        """updates the BHM

        Performs the E-M optimization algorithm for the given (X, y) data. Uses the variational
        parameter xi to lower bound the sigmoid likelihood with a gaussian.

        Args:
            - X: (P, 3)
            - y: (P,) binary array corresponding to occupied (1) and empty (0)
        """
        P = X.shape[0]
        H = self.hinge_points.shape[0] + 1
        y = y.to(self.hinge_points.dtype)
        num_iterations = (
            self.num_iterations
        ) if self.num_iterations_initial is None else self.num_iterations_initial
        self.num_iterations_initial = None
        sigma_inv_0 = self.sigma_inv
        mu_0 = self.mu
        feats = self._get_feats(X)  # (P, H+1)

        # E-M algo
        xi = torch.ones(P, device=self.device, dtype=X.dtype)  # should this be 0?
        for i in range(num_iterations):
            # E step
            sigma_inv = sigma_inv_0 + (feats.T * 2 * torch.abs(_lambda(xi))) @ feats

            mu = torch.linalg.solve(sigma_inv, (sigma_inv_0 @ mu_0 + feats.T @ (y - 0.5)))
            
            # M step
            feats_sig_feats = torch.sum(torch.linalg.solve(sigma_inv.T, feats.T).T * feats, dim=1)
            xi = torch.sqrt(feats_sig_feats + (feats @ mu) ** 2 )
            
        self.sigma_inv = sigma_inv  # (H+1, H+1)
        self.mu = mu  # (H+1,)


    def predict(self, X: Tensor) -> Tensor:
        """returns expected occupancy probability

        Approximates E_w [sigmoid(w^T \phi(x))].
        
        Args:
            - X: (Q, 3) query points
        
        Returns:
            - expected_y: (Q,) expected occupany probability corresponding to query points
        """
        feats = self._get_feats(X)  # (Q, H+1)
        Q, H = feats.shape
        means =  feats @ self.mu # (Q,)
        # variances = (feats.reshape((Q, 1, H)) @ self.sigma @ feats.reshape((Q, H, 1))).reshape(Q)
        variances = torch.sum(torch.linalg.solve(self.sigma_inv.T, feats.T).T * feats, dim=1)
        return torch.sigmoid(means / torch.sqrt(1 + (torch.pi / 8) * variances))  # (Q,)
    
    def to(self, device: torch.device):
        return BayesianHilbertMapWithFullCovarianceMatrixNoInv(
            self.hinge_points.to(device),
            kernel=self.kernel,
            num_iterations=self.num_iterations,
            num_iterations_initial=self.num_iterations_initial,
            sigma_0=self.sigma_inv.to(device),
            mu_0=self.mu.to(device),
            sigma_is_inv=True,
        )
    
    def sequential_update(self, X: Tensor, y: Tensor, max_points_in_update: int) -> None:
        perm = torch.randperm(X.shape[0])
        k = 0
        while k < len(perm):
            idxs = perm[k:k + max_points_in_update]
            self.update(X[idxs], y[idxs])
            k += max_points_in_update



