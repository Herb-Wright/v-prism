import torch
from v_prism.mapping.hinge_point_generation import generate_hingepoint_grid, generate_hingepoint_gaussian


def test_generate_hingepoint_grid():
    expected = torch.tensor([
        [0,  2],
        [0.5,  2],
        [1, 2],
        [1.5, 2],
        [0, 2.5],
        [0.5, 2.5],
        [1, 2.5],
        [1.5, 2.5],
    ])

    grid = generate_hingepoint_grid(
        min=[0, 2],
        max=[2, 3],
        resolution=0.5,
        dtype=expected.dtype,
    )

    assert torch.allclose(expected, grid)


def test_generate_hingepoint_gaussian_no_exception():
    hinge_points = generate_hingepoint_gaussian(
        69,
        [30, 1, 1],
        1.6
    )

    assert hinge_points.shape == (69, 3)



