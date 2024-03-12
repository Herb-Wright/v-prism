import torch
from v_prism.mapping.v_prism import VPRISM
from v_prism.mapping.hinge_point_generation import generate_hingepoint_grid
from v_prism.utils.kernels import GaussianKernel


def test_bhm_forward():
    P = 1000

    hinge_points = generate_hingepoint_grid([0, 0], [1.01, 1.01], 0.2)

    bhm = VPRISM(
        num_classes=2,
        hinge_points=hinge_points,
        kernel=GaussianKernel(50)
    )

    X = torch.rand((1000, 2), dtype=hinge_points.dtype)
    y = (X[:, 0] < 0.5).to(torch.int64)
    
    err_pre = torch.sum(torch.abs(bhm.predict(X)[:, 1] - y))
    bhm.update(X, y)
    err_post = torch.sum(torch.abs(bhm.predict(X)[:, 1] - y))

    assert err_post < err_pre


