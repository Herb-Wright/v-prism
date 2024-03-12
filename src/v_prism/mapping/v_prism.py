
import torch
from torch import Tensor

from ..utils.kernels import Kernel


def _lambda(xi: Tensor) -> Tensor:
    return (0.5 - torch.sigmoid(xi)) / (2*xi)

def VPRISM_update_EM_algorithm(
    sigmas_inv_0: Tensor,
    mus_0: Tensor,
    feats: Tensor,
    labels: Tensor,
    num_iterations: int,
    num_classes: int,
    device: torch.device
) -> tuple[Tensor, Tensor]:
    """returns (sigmas_inv, mus_inv)"""
    feats_y = torch.eye(num_classes, dtype=labels.dtype, device=device)[labels.to(torch.int64)]
    feats_y = feats_y.to(feats.dtype)
    P, H = feats.shape
    alpha_coeff = 0.5 * (0.5 * num_classes - 1)
    # E-M algo
    xi = torch.ones((num_classes, P), device=device, dtype=feats.dtype)
    alpha = torch.zeros(P, device=device, dtype=feats.dtype)

    for i in range(num_iterations):
        lambda_xi = torch.abs(_lambda(xi))  # (K, P)

        # E step
        sigma_inv = sigmas_inv_0 + 2 * feats.T * lambda_xi.reshape((num_classes, 1, P)) @ feats
        sigma: Tensor = torch.cholesky_inverse(torch.linalg.cholesky(sigma_inv))

        mu_times_sig_0_inv = (sigmas_inv_0 @ mus_0.reshape((num_classes, H, 1))).reshape((num_classes, H))  # (K, H)
        untransformed_mu = mu_times_sig_0_inv + (feats_y.T - 0.5 + 2 * alpha * lambda_xi) @ feats  # (K, H)
        mu = (sigma @ untransformed_mu.reshape((num_classes, H, 1))).squeeze()
        
        # M step
        mean_dot_feat = mu @ feats.T  # (K, P)
        alpha = alpha_coeff * torch.sum(lambda_xi * mean_dot_feat, dim=0) / torch.sum(lambda_xi, dim=0)

        feat_sig_feat = torch.sum((feats @ sigma) * feats, dim=2) # (K, P)
        alpha_terms = alpha ** 2 - 2 * alpha * mean_dot_feat
        xi = torch.sqrt(feat_sig_feat + mean_dot_feat ** 2 + alpha_terms)
    return sigma_inv, mu


class VPRISM:
    """VPRISM mapping portion implementation"""
    def __init__(
        self,
        num_classes: int,
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
            - (optional) sigmas_0: (float) or (H+1, H+1) the initial variance matrix/value. defaults to
            1e4.
            - (optional) mus_0: (float) or (H+1,) the initial mean. defaults to 0
        """
        H = hinge_points.shape[0]
        self.device = hinge_points.device
        if isinstance(sigma_0, float) or isinstance(sigma_0, int):
            sigma_0_inv = torch.eye(H + 1, device=self.device, dtype=hinge_points.dtype) / sigma_0
        if isinstance(mu_0, float) or isinstance(mu_0, int):
            mu_0 = torch.ones(H + 1, device=self.device, dtype=hinge_points.dtype) * mu_0
        if sigma_is_inv:
            self.sigmas_inv = sigma_0
            self.mus = mu_0
        else:
            self.sigmas_inv = torch.stack([sigma_0_inv for i in range(num_classes)])  # (K, H, H)
            self.mus = torch.stack([mu_0 for i in range(num_classes)])  # (K, H)
        self.hinge_points = hinge_points
        self.num_iterations = num_iterations
        self.num_iterations_initial = num_iterations_initial
        self.kernel = kernel
        self.num_classes = num_classes

    def _get_feats(self, X: Tensor) -> Tensor:
        N = X.shape[0]
        feats = self.kernel(X, self.hinge_points)
        return torch.concatenate([feats, torch.ones_like(X[:, 0:1])], dim=1)
    
    def update(self, X: Tensor, y: Tensor) -> None:
        """updates the map

        Performs the E-M optimization algorithm for the given (X, y) data. Uses the variational
        parameter xi to lower bound the sigmoid likelihood with a gaussian.

        Args:
            - X: (P, 3)
            - y: (P,) integer array corresponding to occupied object id (>0) or empty (0)
        """
        num_iterations = (
            self.num_iterations
        ) if self.num_iterations_initial is None else self.num_iterations_initial
        self.num_iterations_initial = None
        feats = self._get_feats(X)  # (P, H+1)

        self.sigmas_inv, self.mus = VPRISM_update_EM_algorithm(
            self.sigmas_inv,
            self.mus,
            feats=feats,
            labels=y,
            num_iterations=num_iterations,
            num_classes=self.num_classes,
            device=self.device
        )

    def predict(self, X: Tensor) -> Tensor:
        """
        Args:
            - X: (P, D)
            - num_samples: (int)

        Returns:
            - y: (P, K)
        """
        K = self.num_classes
        feats = self._get_feats(X)
        P, H = feats.shape
        feats_w_mean = feats @ self.mus.T  # (P, K)
        feats_w_variance = torch.sum(torch.linalg.solve(self.sigmas_inv.permute(0, 2, 1), feats.T).permute(0, 2, 1) * feats, dim=2).T  # (P, K)
        
        # NOTE: we assume each class's weights are independent
        z_mean = feats_w_mean.reshape((P, K, 1)) - feats_w_mean.reshape((P, 1, K))  # (P, K, K)
        z_var = feats_w_variance.reshape((P, K, 1)) + feats_w_variance.reshape((P, 1, K))  # should be (P, K, K) of z_k - z_j (k is second dim, j is third)
    
        expected_sigs = torch.sigmoid(z_mean / torch.sqrt(1 + torch.pi * z_var / 8))  # (P, K, K) laplace approx

        K_indices = torch.arange(K, dtype=torch.int64, device=self.device)  # (K,)
        inv_expected_sig_sums = torch.sum(1 / expected_sigs, dim=2) - (1 / expected_sigs[:, K_indices, K_indices])  # (P, K)
        expected_val = 1 / (2 - K + inv_expected_sig_sums)  # (P, K)
        return expected_val



    def to(self, device: torch.device):
        return VPRISM(
            hinge_points=self.hinge_points.to(device),
            kernel=self.kernel,
            num_iterations=self.num_iterations,
            num_iterations_initial=self.num_iterations_initial,
            sigma_0=self.sigmas_inv.to(device),
            mu_0=self.mus.to(device),
            num_classes=self.num_classes,
            sigma_is_inv = True,
        )
    
    def sequential_update(self, X: Tensor, y: Tensor, max_points_in_update: int) -> None:
        perm = torch.randperm(X.shape[0])
        k = 0
        while k < len(perm):
            idxs = perm[k:k + max_points_in_update]
            self.update(X[idxs], y[idxs])
            k += max_points_in_update
    
    def predict_sample(self, X: Tensor, num_samples: int = 30): 
        feats = self._get_feats(X)  # (P, H)
        P, H = feats.shape
        samples = []
        sigmas = torch.cholesky_inverse(torch.linalg.cholesky(self.sigmas_inv))
        for i in range(self.num_classes):
            dist = torch.distributions.MultivariateNormal(self.mus[i], sigmas[i])
            W_i = dist.sample((num_samples,))  # (N, H)
            samples.append(W_i)
        W_sampled = torch.stack(samples, dim=0)  # (K, N, H)
        feats = W_sampled @ feats.T  # (K, N, P)
        feats = feats.permute((1, 2, 0))  # (N, P, K)
        probits = torch.softmax(feats, dim=-1)
        return probits





