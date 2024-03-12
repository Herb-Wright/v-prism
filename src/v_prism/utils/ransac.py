"""
Implementation of RANSAC algorithm for finding a plane
"""
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Plane:
    """plane defined by 0 = p.bias + p.normal_vect^T x"""
    normal_vect: Tensor
    bias: float

    def __str__(self) -> str:
        return f'Plane(normal_vect={self.normal_vect}, bias={self.bias})'

def project_onto_plane(plane: Plane, points: Tensor) -> Tensor:
    proj = (((points @ plane.normal_vect) + plane.bias) / (plane.normal_vect @ plane.normal_vect)).reshape(-1, 1) * plane.normal_vect
    return points - proj

def dist_to_plane(plane: Plane, x: Tensor) -> Tensor:
    """
    Args:
        - plane: (Plane) the 3D plane
        - x: (P, 3) the points

    Returns:
        - dists: (P,) the distances corresponding to each point from the given plane
    """
    normal_vect_norm = torch.norm(plane.normal_vect)
    bias_adj = plane.bias / normal_vect_norm
    normal_vect_adj = plane.normal_vect / normal_vect_norm
    return torch.abs(bias_adj + x @ normal_vect_adj)

def plane_through_points(points: Tensor) -> Plane:
    """
    Args:
        - points: (3, 3)
    
    Returns:
        - plane: (Plane) plane that minimizes distance from points
    """
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    diff1 = p1 - p3
    diff2 = p2 - p3
    cross_prod = torch.cross(diff1, diff2)
    val = cross_prod @ p3
    return Plane(
        cross_prod,
        -val
    )


def ransac_plane(points: Tensor, iter: int, dist_tol: float) -> Plane:
    """Runs ransac for fixed number of iterations and returns best result

    Args:
        - points: (P, 3)
        - iter: (int) number of iterations to run ransac
        - dist_tol: (float) the tolerance for distance to plane
    
    Returns:
        - plane: (Plane)
    """
    P, D = points.shape
    best_num = 0
    best_plane = None
    for i in range(iter):
        points_chosen = points[torch.randperm(P)[:D]]  # (D, D)
        plane = plane_through_points(points_chosen)
        dists = dist_to_plane(plane, points)
        num = torch.sum(dists <= dist_tol)
        if num > best_num:
            best_num = num
            best_plane = plane
    return best_plane


def robust_ransac(points: Tensor, seg_mask: Tensor, iter: int, dist_tol: float, radius: float) -> Plane:
    scene_center = 0.5 * (torch.amax(points[seg_mask > 0], dim=0) + torch.amin(points[seg_mask > 0], dim=0))
    dists_from_center = torch.norm(points - scene_center, dim=1)
    filtered_points = points[torch.logical_and(dists_from_center <= radius, seg_mask == 0)]
    # (1) run ransac
    plane = ransac_plane(filtered_points, iter, dist_tol)
    # (2) flip to correct orientation
    center_dist_from_plane_sgn = plane.bias + torch.dot(scene_center, plane.normal_vect)
    if center_dist_from_plane_sgn < 0:
        plane.bias = -plane.bias
        plane.normal_vect = -plane.normal_vect
    return plane





