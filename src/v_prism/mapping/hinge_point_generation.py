import torch
from torch import Tensor
import numpy as np


def generate_hingepoint_grid(
    min: list[float], 
    max: list[float], 
    resolution: float,
    *,
    dtype: torch.dtype | None = None
) -> Tensor:
    """generates hingepoints in a grid
    
    Args:
        - min: (list[float]) a list of D floats
        - max: (list[float]) a list of D floats
        - resolution: (float)

    Returns:
        - hinge_points: (H, D)
    """
    assert len(max) == len(min)
    D = len(max)
    ranges = [np.arange(min[i], max[i], step=resolution) for i in range(D)]
    grid: Tensor = torch.tensor(np.stack(np.meshgrid(*ranges), axis=-1))
    if dtype is not None:
        grid = grid.to(dtype)
    return grid.reshape((-1, D))

def generate_hingepoint_gaussian(k: int, mean: Tensor | list[float], std: float) -> Tensor:
    """generates hingepoints from normal distribution
    
    Args:
        - k: (int) number of samples H
        - mean: (D,) or (list[float]) mean of samples
        - std: (float) the standard deviation of the samples 
    
    Returns:
        - hinge_points: (H, D)
    """
    return torch.randn(k, len(mean)) * std + torch.tensor(mean)


def generate_hingepoint_grid_for_multiple_objects(
    points: Tensor, 
    segmentation: Tensor, 
    resolution: float,
    dist_from_obj: float,
    *,
    dtype: torch.dtype | None = None
) -> Tensor:
    """generates hingepoint grid for multiple objects

    Filters hingepoints based on if they are <= dist_from_obj from the median of an object.

    Args:
        - points: (P, D)
        - segmentation: (P,) array of integers
        - resolution: (float) the length between consecutive hingepoints
        - dist_from_obj: (float) the max dist from hingepoint to one of the object's points
    
    Returns:
        - hinge_points: (H, D)
    """
    D = points.shape[1]
    max_points = torch.amax(points[segmentation > 0], dim=0) + dist_from_obj + resolution
    min_points = torch.amin(points[segmentation > 0], dim=0) - dist_from_obj
    ranges = [np.arange(min_points[i], max_points[i], step=resolution) for i in range(D)]
    grid: Tensor = torch.tensor(np.stack(np.meshgrid(*ranges), axis=-1)).reshape((-1, D))
    H_pre = grid.shape[0]
    if dtype is not None:
        grid = grid.to(dtype)
    ids = torch.unique(segmentation[segmentation > 0])
    centers = []
    for id in ids:
        object_min = torch.amin(points[segmentation == id], dim=0)
        object_max = torch.amax(points[segmentation == id], dim=0)
        centers.append(0.5 * (object_max + object_min))
    object_centers = torch.stack(centers, dim=0) # (N, D)
    N = object_centers.shape[0]
    dists = torch.cdist(object_centers.reshape((1, N, D)), grid.reshape((1, H_pre, D))).reshape((N, H_pre))  # (N, H_pre)
    min_dists = torch.amin(dists, dim=0)  # (H_pre)
    selected_grid = grid[min_dists <= dist_from_obj]  # filter far points
    return selected_grid
    

def generate_hingepoint_grid_multiple_objects_w_surface(
    points: Tensor, 
    segmentation: Tensor, 
    resolution: float,
    dist_from_obj: float,
    num_surface_points_per_obj: int = 16,
    *,
    dtype: torch.dtype | None = None
) -> Tensor:
    """generates hinge points consisting of a grid and some surface points

    Args:
        - points: (P, 3) the points
        - segmentation: (P,) integer array that denotes the seg id of each point
        - resolution: (float) the distance between hinge points in grid
        - dist_from_obj: (float) the maximum distance from objects for the hingepoints to be
        - num_surface_points_per_obj: (int) num of surface points sampled of each object
        - dtype: (torch.dtype) the dtype of the hingepoints 
    
    Returns:
        - hinge_points: (H, 3)
    """
    grid_hinge_points = generate_hingepoint_grid_for_multiple_objects(
        points=points,
        segmentation=segmentation,
        resolution=resolution,
        dist_from_obj=dist_from_obj,
        dtype=dtype
    )  # (H', 3)
    hps = [grid_hinge_points]
    num_objects = int(torch.amax(segmentation).item())
    for id in range(1, num_objects + 1):
        surface_points = points[segmentation == id]
        idxs = torch.randperm(surface_points.shape[0])[:num_surface_points_per_obj]
        hps.append(surface_points[idxs])
    hinge_points = torch.concat(hps, dim=0)  # (H, 3)
    return hinge_points



