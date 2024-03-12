import os
from typing import Callable

import torch
from torch import Tensor, from_numpy
import point_cloud_utils as pcu
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform.rotation import Rotation
import trimesh

from .visualization import gen_mesh_for_sdf_batch_3d
from ..mapping.hinge_point_generation import generate_hingepoint_grid


def calc_iou_trimesh(
    mesh_path: str, 
    pred_func: Callable[[Tensor], Tensor], 
    resolution: float,
    *,
    conf: float = 0.5,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    max_elem_in_func_call: int = 10_000
) -> float:
    mesh = _as_mesh(trimesh.load(mesh_path))
    mesh.vertices = _transform_array(mesh.vertices, mesh_position, mesh_orientation, mesh_scale)

    mins = torch.amin(from_numpy(mesh.vertices), dim=0)
    maxs = torch.amax(from_numpy(mesh.vertices), dim=0)
    centroid = 0.5 * (maxs + mins)
    
    # 1. create input points
    grid = generate_hingepoint_grid(centroid - BUFFER, centroid + BUFFER, resolution=resolution)
    fn1_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    fn2_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    # 2. evaluate fn1 and fn2
    perm = torch.randperm(grid.shape[0])
    k = 0
    while k < len(perm):
        idxs = perm[k:k + max_elem_in_func_call]
        fn1_preds[idxs] = pred_func(grid[idxs]) > conf
        fn2_preds[idxs] = from_numpy(mesh.contains(grid[idxs])) > conf
        k += max_elem_in_func_call
    
    # 3. calc iou
    intersection = torch.sum(torch.logical_and(fn1_preds, fn2_preds))
    union = torch.sum(torch.logical_or(fn1_preds, fn2_preds))
    if union == 0:
        return 0.0
    return intersection / union

def calc_iou(
    mesh_path: str, 
    pred_func: Callable[[Tensor], Tensor], 
    resolution: float,
    *,
    conf: float = 0.5,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    max_elem_in_func_call: int = 10_000
) -> float:
    v, f = pcu.load_mesh_vf(mesh_path)
    v, f = pcu.make_mesh_watertight(v, f, 4_000)
    v = _transform_array(v, mesh_position, mesh_orientation, mesh_scale)

    mins = torch.amin(from_numpy(v), dim=0)
    maxs = torch.amax(from_numpy(v), dim=0)
    centroid = 0.5 * (maxs + mins)
    
    # 1. create input points
    grid = generate_hingepoint_grid(centroid - BUFFER, centroid + BUFFER, resolution=resolution)
    fn1_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    fn2_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    # 2. evaluate fn1 and fn2
    perm = torch.randperm(grid.shape[0])
    k = 0
    while k < len(perm):
        idxs = perm[k:k + max_elem_in_func_call]
        fn1_preds[idxs] = pred_func(grid[idxs]) > conf
        gt_sdf_vals, _, _  = pcu.signed_distance_to_mesh(grid[idxs].numpy(), v, f)
        fn2_preds[idxs] = from_numpy(gt_sdf_vals) < 0
        k += max_elem_in_func_call
    
    # 3. calc iou
    intersection = torch.sum(torch.logical_and(fn1_preds, fn2_preds))
    union = torch.sum(torch.logical_or(fn1_preds, fn2_preds))
    if union == 0:
        return 0.0
    return intersection / union


BUFFER = 0.2

def calc_soft_iou(
    mesh_path: str, 
    pred_func: Callable[[Tensor], Tensor], 
    resolution: float,
    *,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    max_elem_in_func_call: int = 10_000
) -> float:
    mesh = _as_mesh(trimesh.load(mesh_path))
    mesh.vertices = _transform_array(mesh.vertices, mesh_position, mesh_orientation, mesh_scale)

    mins = torch.amin(from_numpy(mesh.vertices), dim=0)
    maxs = torch.amax(from_numpy(mesh.vertices), dim=0)
    centroid = 0.5 * (maxs + mins)
    print(centroid)
    
    # 1. create input points
    grid = generate_hingepoint_grid(centroid - BUFFER, centroid + BUFFER, resolution=resolution)
    fn1_preds = torch.zeros_like(grid[:, 0])
    fn2_preds = torch.zeros_like(grid[:, 0])

    # 2. evaluate fn1 and fn2
    perm = torch.randperm(grid.shape[0])
    k = 0
    while k < len(perm):
        idxs = perm[k:k + max_elem_in_func_call]
        fn1_preds[idxs] = pred_func(grid[idxs])
        fn2_preds[idxs] = from_numpy(mesh.contains(grid[idxs])).to(fn2_preds.dtype)
        k += max_elem_in_func_call
    
    # 3. calc iou
    intersection = torch.sum(torch.minimum(fn1_preds, fn2_preds))
    union = torch.sum(torch.maximum(fn1_preds, fn2_preds))
    if union == 0:
        return 0.0
    return intersection / union


def chamfer_dist_for_mesh(
    mesh_path: str, 
    pred_func: Callable[[Tensor], Tensor], 
    resolution: float,
    *,
    conf: float = 0.5,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    sample_count: int = 10_000,
    device: torch.device = torch.device("cpu")
) -> float:
    # (1) load mesh
    v, f = pcu.load_mesh_vf(mesh_path)
    v = _transform_array(v, np.array(mesh_position), np.array(mesh_orientation), mesh_scale)
    vm, fm = pcu.make_mesh_watertight(v, f, 10_000)
    # (2) calc counts
    mins = torch.amin(from_numpy(v), dim=0)
    maxs = torch.amax(from_numpy(v), dim=0)
    centroid = 0.5 * (maxs + mins)
    # construct mesh
    recon_mesh = gen_mesh_for_sdf_batch_3d(
        pred_func, 
        xlim=[centroid[0] - BUFFER, centroid[0] + BUFFER], 
        ylim=[centroid[1] - BUFFER, centroid[1] + BUFFER], 
        zlim=[centroid[2] - BUFFER, centroid[2] + BUFFER], 
        resolution=resolution, 
        confidence=conf, 
        device=device
    )

    if recon_mesh is None:
        return torch.nan  # what to do here?
    recon_samples = recon_mesh.sample(sample_count)
    fid, bc = pcu.sample_mesh_random(vm, fm, sample_count)
    gt_samples = pcu.interpolate_barycentric_coords(fm, fid, bc, vm)
    dist = pcu.chamfer_distance(recon_samples, gt_samples)
    return dist

def _transform_array(
    points: NDArray, 
    translation: NDArray, 
    quat: NDArray,
    scale: float = 1,
) -> NDArray:
    my_rotation: Rotation = Rotation.from_quat(quat)
    new_points = points * scale
    new_points: NDArray = my_rotation.apply(new_points)
    new_points = new_points + translation
    return new_points


def _as_mesh(scene_or_mesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
        assert(isinstance(mesh, trimesh.Trimesh))
    return mesh

