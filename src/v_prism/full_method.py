import torch 
from torch import Tensor

from .utils.kernels import GaussianKernel
from .data_loading.negative_sampling import full_negative_sampling_method
from .mapping.hinge_point_generation import generate_hingepoint_grid_multiple_objects_w_surface
from .mapping.v_prism import VPRISM, VPRISM_update_EM_algorithm


def full_VPRISM_method(
    points: Tensor,
    seg_mask: Tensor,
    num_classes: int,
    camera_pos: Tensor,
    *,
    grid_len=0.05,
    grid_dist_from_obj=0.15,
    kernel_param=1000,
    ray_step_size=0.1,
    object_sphere_radius=0.25,
    subsample_grid_size_occ=0.01,
    subsample_grid_size_unocc=0.015,
    num_surface_points_per_obj=32,
    scene_sphere_radius=0.4,
    device=torch.device("cpu"),
    max_points_in_update=20000,
) -> VPRISM:
    hinge_points = generate_hingepoint_grid_multiple_objects_w_surface(
        points, 
        seg_mask, 
        grid_len, 
        grid_dist_from_obj, 
        num_surface_points_per_obj, 
        dtype=points.dtype,
    ).to(device)
    kernel = GaussianKernel(kernel_param)
    map = VPRISM(num_classes, hinge_points, kernel, num_iterations=3).to(device)
    X, y = full_negative_sampling_method(
        points,
        seg_mask,
        ray_step_size,
        object_sphere_radius,
        camera_pos,
        scene_sphere_radius,
        subsample_grid_size_unocc,
        subsample_grid_size_occ,
    )
    map.sequential_update(X.to(device), y.to(device), max_points_in_update=max_points_in_update)
    return map

