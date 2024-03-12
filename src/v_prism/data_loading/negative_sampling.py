
import torch
from torch import Tensor

from .scene import Scene
from ..utils.constants import Labels
from ..utils.ransac import robust_ransac
from ..utils.subsample import grid_subsample_different_res


def negative_sample_each_ray_for_ablation(points: Tensor, seg_mask: Tensor, camera_pos: Tensor) -> tuple[Tensor, Tensor]:
    """samples a negative point along the ray for each point. 
    
    Used in ablation experiment on negative sampling.
    
    Args:
        - points: (P', 3)
        - seg_mask: (P')
        - camera_pos: (3,)

    Returns:
        - X: (P, 3)
        - y: (P,)
    """
    convex_comb = torch.rand((points.shape[0], 1))  # (P',)
    negative_points = convex_comb * points + (1 - convex_comb) * camera_pos
    X = torch.concatenate([points, negative_points], axis=0)
    y = torch.concatenate([seg_mask, torch.zeros_like(negative_points[:, 0])], axis=0)
    return X, y


def add_negative_points_below_plane(
    X: Tensor, 
    y: Tensor, 
    plane: list[float],
    center: Tensor, 
    radius: float,
    k: int = 1000
) -> tuple[Tensor, Tensor]:
    unit_samples = 2 * torch.rand(size=(k, len(center)), dtype=X.dtype, device=X.device) - 1
    mask1 = torch.sum(unit_samples ** 2, axis=1) <= 1
    sphere_samples = center + radius * unit_samples[mask1]
    normal_vect = torch.tensor([plane[0], plane[1], plane[2]], dtype=X.dtype, device=X.device)
    thresh = plane[3]
    mask_plane = sphere_samples @ normal_vect < thresh
    if torch.sum(mask_plane) == 0:
        return X, y  # early exit if no negative points found
    samples = sphere_samples[mask_plane]
    X_new = torch.concatenate([X, samples], axis=0)
    y_new = torch.concatenate([y, Labels.EMPTY * torch.ones_like(samples[:, 0])], axis=0)
    return X_new, y_new
    

def negative_sample_rays_in_sphere_uniform_each_step_multiclass(
    points: Tensor,
    mask: Tensor,
    step_size: float,
    radius: float,
    *,
    camera_pos: Tensor = torch.zeros(3),
    max_points: int = 1_000_000
) -> tuple[Tensor, Tensor]:
    """negatively samples points along rays within a sphere around the object

    Args:
        - points: (P', 3) the pointcloud
        - mask: (P',) the integer mask of the objects where 0 is background and >0 is object id 
        - step_size: (float) the step size to step forward
        - radius: (float) radius of the ball around the centroid of the points within the mask
        - (optional) camera_pos: (3,) position of the camera

    Returns: a tuple,
        - X: (P, 3) features
        - y: (P,) labels of 1s and 0s
    """
    if torch.sum(mask > 0) <= 0:
        raise Exception("must have more than one point in the mask to be part of an object")
    obj_points = points[mask > 0]
    K = len(torch.unique(mask[mask > 0]))
    centers = []
    for i in torch.unique(mask[mask > 0]):
        centers.append(0.5 * (torch.amin(points[mask == i], dim=0) + torch.amax(points[mask == i], dim=0)))
    centroids = torch.stack(centers) # (K, 3)
    centroid_dist = torch.linalg.norm(centroids - camera_pos, dim=1)  # (K,)
    min_dist = torch.amin(centroid_dist) - radius
    max_dist = torch.amax(centroid_dist) + radius
    ray_vects =  points - camera_pos
    depth: Tensor = torch.linalg.norm(ray_vects, axis=1)
    rays: Tensor = ray_vects / depth.reshape((-1, 1))
    P_poss = rays.shape[0]
    dists = torch.arange(min_dist, max_dist, step=step_size)  # (R,)
    R = dists.shape[0]
    dists_noise = dists.reshape((-1, 1)) + torch.rand((R, P_poss)) * step_size
    possible_neg_points = camera_pos + dists_noise.reshape((R, P_poss, 1)) * rays  # (R, P~, 3)
    depth_mask = dists_noise < depth
    dist_mask = torch.amin(torch.linalg.norm(possible_neg_points - centroids.reshape((K, 1, 1, 3)), dim=3), dim=0) <= radius
    mask2 = torch.logical_and(depth_mask, dist_mask)
    negative_points = possible_neg_points[mask2]
    P_neg = negative_points.shape[0]
    if P_neg > max_points:
        idxs = torch.randperm(P_neg)[:max_points]
        negative_points = negative_points[idxs]
    
    X = torch.concatenate([obj_points, negative_points], axis=0)

    y = torch.concatenate([
        mask[mask > 0],  # object id
        torch.ones_like(negative_points[:, 0]) * Labels.EMPTY,  # 0
    ])

    return X, y

def negative_sample_rays_in_sphere_step_multiclass(
    points: Tensor,
    mask: Tensor,
    step_size: float,
    radius: float,
    *,
    camera_pos: Tensor = torch.zeros(3),
    max_points: int = 1_000_000
) -> tuple[Tensor, Tensor]:
    """negatively samples points along rays within a sphere around the object

    Args:
        - points: (P', 3) the pointcloud
        - mask: (P',) the integer mask of the objects where 0 is background and >0 is object id 
        - step_size: (float) the step size to step forward
        - radius: (float) radius of the ball around the centroid of the points within the mask
        - (optional) camera_pos: (3,) position of the camera

    Returns: a tuple,
        - X: (P, 3) features
        - y: (P,) labels of 1s and 0s
    """
    if torch.sum(mask > 0) <= 0:
        raise Exception("must have more than one point in the mask to be part of an object")
    obj_points = points[mask > 0]
    K = len(torch.unique(mask[mask > 0]))
    centers = []
    for i in torch.unique(mask[mask > 0]):
        centers.append(0.5 * (torch.amin(points[mask == i], dim=0) + torch.amax(points[mask == i], dim=0)))
    centroids = torch.stack(centers) # (K, 3)
    centroid_dist = torch.linalg.norm(centroids - camera_pos, dim=1)  # (K,)
    min_dist = torch.amin(centroid_dist) - radius
    max_dist = torch.amax(centroid_dist) + radius
    ray_vects =  points - camera_pos
    depth: Tensor = torch.linalg.norm(ray_vects, axis=1)
    rays: Tensor = ray_vects / depth.reshape((-1, 1))
    P_poss = rays.shape[0]
    dists = torch.arange(min_dist, max_dist, step=step_size)  # (R,)
    R = dists.shape[0]
    dists_noise = dists.reshape((-1, 1)) + torch.zeros((R, P_poss), dtype=dists.dtype, device=dists.device)
    possible_neg_points = camera_pos + dists_noise.reshape((R, P_poss, 1)) * rays  # (R, P~, 3)
    depth_mask = dists_noise < depth
    dist_mask = torch.amin(torch.linalg.norm(possible_neg_points - centroids.reshape((K, 1, 1, 3)), dim=3), dim=0) <= radius
    mask2 = torch.logical_and(depth_mask, dist_mask)
    negative_points = possible_neg_points[mask2]
    P_neg = negative_points.shape[0]
    if P_neg > max_points:
        idxs = torch.randperm(P_neg)[:max_points]
        negative_points = negative_points[idxs]
    
    X = torch.concatenate([obj_points, negative_points], axis=0)

    y = torch.concatenate([
        mask[mask > 0],  # object id
        torch.ones_like(negative_points[:, 0]) * Labels.EMPTY,  # 0
    ])

    return X, y

def add_negative_points_below_plane_multiple_objects(
    X: Tensor, 
    y: Tensor, 
    plane: list[float],
    center: Tensor, 
    radius: float,
    k: int = 100_000
) -> tuple[Tensor, Tensor]:
    """adds points below plane in sphere around each object

    Args:
        - X: (P', 3) point cloud
        - y: (P',) segmentation labels for point cloud
        - plane: (list[float]) specifies the plane plane[3] = plane[:3]^T x
        - center: (3,) scene center
        - radius: (float) radius from object center to keep points from
        - k: (int) number of potential samples

    Returns: (tuple)
        - X_new: (P, 3)
        - y_new: (P,)
    """
    centers = []
    for i in torch.unique(y[y > 0]):
        centers.append(0.5 * (torch.amin(X[y == i], dim=0) + torch.amax(X[y == i], dim=0)))
    centroids = torch.stack(centers) # (N, 3)
    N, D = centroids.shape
    unit_samples = 2 * torch.rand(size=(k, len(center)), dtype=X.dtype, device=X.device) - 1
    mask1 = torch.sum(unit_samples ** 2, axis=1) <= 1
    sphere_samples = (centroids.reshape((N, 1, D)) + radius * unit_samples[mask1]).reshape((-1, D))
    normal_vect = torch.tensor([plane[0], plane[1], plane[2]], dtype=X.dtype, device=X.device)
    thresh = plane[3]
    mask_plane = (sphere_samples @ normal_vect) < thresh
    if torch.sum(mask_plane) == 0:
        sphere_samples
        return X, y  # early exit if no negative points found
    samples = sphere_samples[mask_plane]
    X_new = torch.concatenate([X, samples], axis=0)
    y_new = torch.concatenate([y, Labels.EMPTY * torch.ones_like(samples[:, 0])], axis=0)
    return X_new, y_new


def full_negative_sampling_method(
    points,
    seg_mask,
    ray_step_size,
    object_sphere_radius,
    camera_pos,
    scene_sphere_radius,
    subsample_grid_size_unocc,
    subsample_grid_size_occ,
) -> tuple[Tensor, Tensor]:
    scene_center = 0.5 * (
        torch.amax(points[seg_mask > 0], dim=0) 
        + torch.amin(points[seg_mask > 0], dim=0)
    )
    X, y = negative_sample_rays_in_sphere_uniform_each_step_multiclass(
        points=points, 
        mask=seg_mask, 
        step_size=ray_step_size, 
        radius=object_sphere_radius, 
        camera_pos=camera_pos,
    )
    plane = robust_ransac(points, seg_mask, 100, dist_tol=0.01, radius=scene_sphere_radius)
    X, y = add_negative_points_below_plane_multiple_objects(
        X, y, plane=[*list(plane.normal_vect), -plane.bias], center=scene_center, radius=object_sphere_radius, k=1000
    )
    X, y = grid_subsample_different_res(
        X, 
        y, 
        subsample_grid_size_unocc=subsample_grid_size_unocc, 
        subsample_grid_size_occ=subsample_grid_size_occ
    )
    return X, y

