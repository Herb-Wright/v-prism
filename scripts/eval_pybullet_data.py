from argparse import ArgumentParser
import os
import time

import torch
from torch import Tensor, from_numpy
import numpy as np

from v_prism.data_loading.bullet_reconstruction_dataset import ReconBulletDataset
from v_prism.data_loading.scene import Scene
from v_prism.mapping.hinge_point_generation import generate_hingepoint_grid_multiple_objects_w_surface
from v_prism.mapping.v_prism import VPRISM
from v_prism.mapping.bayesian_hilbert_map import BayesianHilbertMapWithFullCovarianceMatrixNoInv
from v_prism.utils.kernels import GaussianKernel
from v_prism.data_loading.negative_sampling import negative_sample_rays_in_sphere_uniform_each_step_multiclass
from v_prism.data_loading.negative_sampling import negative_sample_rays_in_sphere_step_multiclass
from v_prism.data_loading.negative_sampling import add_negative_points_below_plane_multiple_objects
from v_prism.data_loading.negative_sampling import negative_sample_each_ray_for_ablation
from v_prism.utils.ransac import robust_ransac
from v_prism.utils.subsample import grid_subsample, grid_subsample_different_res
from v_prism.utils.ui import abspath
from v_prism.utils.eval import calc_iou, chamfer_dist_for_mesh
from v_prism.utils.pointsdf import PointSDF, scale_and_center_object_points
from v_prism.utils.pointsdf import scale_and_center_queries, index_points
from v_prism.utils.pointsdf import  farthest_point_sample


# (1) arg parse
parser = ArgumentParser()
parser.add_argument("-o", "--output", type=str, default=None)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-a", "--algorithm", type=str)
parser.add_argument("--camera_at_origin", action="store_true")
args = parser.parse_args()

# (2) helper funcs/stuff
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"using device {device}!")
data_dir = abspath("~/data")
logs_dir = abspath(os.path.join(os.path.dirname(__file__), "../logs"))

RESOLUTION = 0.015
EPSILON = 1e-7

def print_and_maybe_log(msg: str, *, filename: str | None = None) -> None:
    print(msg)
    if filename is not None:
        with open(filename, "a") as f:
            f.write(msg + "\n")

class SoftmaxBHMReconAlgorithm:
    def __init__(
        self, 
        grid_len: float, 
        grid_dist_from_obj: float, 
        kernel_param: float,
        ray_step_size: float,
        object_sphere_radius: float,
        scene_sphere_radius: float,
        subsample_grid_size_occ: float,
        subsample_grid_size_unocc: float,
        num_surface_points_per_obj: int,
        device: torch.device = device
    ) -> None:
        self.grid_len = grid_len
        self.grid_dist_from_obj = grid_dist_from_obj
        self.kernel_param = kernel_param
        self.ray_step_size = ray_step_size
        self.object_sphere_radius = object_sphere_radius
        self.scene_sphere_radius = scene_sphere_radius
        self.subsample_grid_size_occ = subsample_grid_size_occ
        self.subsample_grid_size_unocc = subsample_grid_size_unocc
        self.device = device
        self.num_surface_points = num_surface_points_per_obj
        self.conf = 0.5
        self.bhm = None

    def fit(self, scene: Scene) -> None:
        hinge_points = generate_hingepoint_grid_multiple_objects_w_surface(
            scene.points, scene.seg_mask, self.grid_len, self.grid_dist_from_obj, self.num_surface_points
        )
        print(f"hinge_points: {hinge_points.shape}")
        kernel = GaussianKernel(self.kernel_param)
        # (2) create bhm
        bhm = VPRISM(
            num_classes=int(torch.amax(scene.seg_mask).item()) + 1,
            hinge_points=hinge_points,
            kernel=kernel,
            num_iterations=3,
            # num_iterations_initial=10
            # num_iterations=5,
        ).to(self.device)
        # (3) neg sampling on data
        scene_center = 0.5 * (
            torch.amax(scene.points[scene.seg_mask > 0], dim=0) 
            + torch.amin(scene.points[scene.seg_mask > 0], dim=0)
        )
        X, y = negative_sample_rays_in_sphere_uniform_each_step_multiclass(
            points=scene.points, 
            mask=scene.seg_mask, 
            step_size=self.ray_step_size, 
            radius=self.object_sphere_radius, 
            camera_pos=scene.camera_pos,
        )
        plane = robust_ransac(scene.points, scene.seg_mask, 100, dist_tol=0.01, radius=self.scene_sphere_radius)
        X, y = add_negative_points_below_plane_multiple_objects(
            X, y, plane=[*list(plane.normal_vect), -plane.bias], center=scene_center, radius=self.object_sphere_radius, k=1000
        )
    
        X, y = grid_subsample_different_res(
            X, 
            y, 
            subsample_grid_size_unocc=self.subsample_grid_size_unocc, 
            subsample_grid_size_occ=self.subsample_grid_size_occ
        )
        print(f"X: {X.shape}")
        # (4) fit bhm
        bhm.sequential_update(X.to(self.device), y.to(self.device), 20000)
        self.bhm = bhm

    def predict(self, x: Tensor) -> Tensor:
        return self.bhm.predict(x.to(self.device)).to(torch.device("cpu"))
    
class SoftmaxBHMReconNoUnderTableAlgorithm:
    def __init__(
        self, 
        grid_len: float, 
        grid_dist_from_obj: float, 
        kernel_param: float,
        ray_step_size: float,
        object_sphere_radius: float,
        scene_sphere_radius: float,
        subsample_grid_size_occ: float,
        subsample_grid_size_unocc: float,
        num_surface_points_per_obj: int,
        device: torch.device = device
    ) -> None:
        self.grid_len = grid_len
        self.grid_dist_from_obj = grid_dist_from_obj
        self.kernel_param = kernel_param
        self.ray_step_size = ray_step_size
        self.object_sphere_radius = object_sphere_radius
        self.scene_sphere_radius = scene_sphere_radius
        self.subsample_grid_size_unocc = subsample_grid_size_unocc
        self.subsample_grid_size_occ = subsample_grid_size_occ
        self.device = device
        self.num_surface_points = num_surface_points_per_obj
        self.conf = 0.5
        self.bhm = None

    def fit(self, scene: Scene) -> None:
        hinge_points = generate_hingepoint_grid_multiple_objects_w_surface(
            scene.points, scene.seg_mask, self.grid_len, self.grid_dist_from_obj, self.num_surface_points
        )
        print(f"hinge_points: {hinge_points.shape}")
        kernel = GaussianKernel(self.kernel_param)
        # (2) create bhm
        bhm = VPRISM(
            num_classes=int(torch.amax(scene.seg_mask).item()) + 1,
            hinge_points=hinge_points,
            kernel=kernel,
            num_iterations=3,
            # num_iterations_initial=10
            # num_iterations=5,
        ).to(self.device)
        # (3) neg sampling on data
        scene_center = 0.5 * (
            torch.amax(scene.points[scene.seg_mask > 0], dim=0) 
            + torch.amin(scene.points[scene.seg_mask > 0], dim=0)
        )
        X, y = negative_sample_rays_in_sphere_uniform_each_step_multiclass(
            points=scene.points, 
            mask=scene.seg_mask, 
            step_size=self.ray_step_size, 
            radius=self.object_sphere_radius, 
            camera_pos=scene.camera_pos,
        )
        X, y = grid_subsample_different_res(
            X, 
            y, 
            subsample_grid_size_unocc=self.subsample_grid_size_unocc, 
            subsample_grid_size_occ=self.subsample_grid_size_occ
        )
        print(f"X: {X.shape}")
        # (4) fit bhm
        bhm.sequential_update(X.to(self.device), y.to(self.device), 20000)
        self.bhm = bhm

    def predict(self, x: Tensor) -> Tensor:
        return self.bhm.predict(x.to(self.device)).to(torch.device("cpu"))


class SoftmaxBHMReconNoStratifiedAlgorithm:
    def __init__(
        self, 
        grid_len: float, 
        grid_dist_from_obj: float, 
        kernel_param: float,
        ray_step_size: float,
        object_sphere_radius: float,
        scene_sphere_radius: float,
        subsample_grid_size_occ: float,
        subsample_grid_size_unocc: float,
        num_surface_points_per_obj: int,
        device: torch.device = device
    ) -> None:
        self.grid_len = grid_len
        self.grid_dist_from_obj = grid_dist_from_obj
        self.kernel_param = kernel_param
        self.ray_step_size = ray_step_size
        self.object_sphere_radius = object_sphere_radius
        self.scene_sphere_radius = scene_sphere_radius
        self.subsample_grid_size_unocc = subsample_grid_size_unocc
        self.subsample_grid_size_occ = subsample_grid_size_occ
        self.device = device
        self.num_surface_points = num_surface_points_per_obj
        self.conf = 0.5
        self.bhm = None

    def fit(self, scene: Scene) -> None:
        hinge_points = generate_hingepoint_grid_multiple_objects_w_surface(
            scene.points, scene.seg_mask, self.grid_len, self.grid_dist_from_obj, self.num_surface_points
        )
        print(f"hinge_points: {hinge_points.shape}")
        kernel = GaussianKernel(self.kernel_param)
        # (2) create bhm
        bhm = VPRISM(
            num_classes=int(torch.amax(scene.seg_mask).item()) + 1,
            hinge_points=hinge_points,
            kernel=kernel,
            num_iterations=3,
            # num_iterations_initial=10
            # num_iterations=5,
        ).to(self.device)
        # (3) neg sampling on data
        scene_center = 0.5 * (
            torch.amax(scene.points[scene.seg_mask > 0], dim=0) 
            + torch.amin(scene.points[scene.seg_mask > 0], dim=0)
        )
        X, y = negative_sample_rays_in_sphere_step_multiclass(
            points=scene.points, 
            mask=scene.seg_mask, 
            step_size=self.ray_step_size, 
            radius=self.object_sphere_radius, 
            camera_pos=scene.camera_pos,
        )
        plane = robust_ransac(scene.points, scene.seg_mask, 100, dist_tol=0.01, radius=self.scene_sphere_radius)
        X, y = add_negative_points_below_plane_multiple_objects(
            X, y, plane=[*list(plane.normal_vect), -plane.bias], center=scene_center, radius=self.object_sphere_radius, k=1000
        )
    
        X, y = grid_subsample_different_res(
            X, 
            y, 
            subsample_grid_size_unocc=self.subsample_grid_size_unocc, 
            subsample_grid_size_occ=self.subsample_grid_size_occ
        )
        print(f"X: {X.shape}")
        # (4) fit bhm
        bhm.sequential_update(X.to(self.device), y.to(self.device), 20000)
        self.bhm = bhm

    def predict(self, x: Tensor) -> Tensor:
        return self.bhm.predict(x.to(self.device)).to(torch.device("cpu"))
    

class SoftmaxBHMReconAlgorithmBadSampling:
    def __init__(
        self, 
        grid_len: float, 
        grid_dist_from_obj: float, 
        kernel_param: float,
        ray_step_size: float,
        object_sphere_radius: float,
        scene_sphere_radius: float,
        subsample_grid_size: float,
        num_surface_points_per_obj: int,
        device: torch.device = device
    ) -> None:
        self.grid_len = grid_len
        self.grid_dist_from_obj = grid_dist_from_obj
        self.kernel_param = kernel_param
        self.ray_step_size = ray_step_size
        self.object_sphere_radius = object_sphere_radius
        self.scene_sphere_radius = scene_sphere_radius
        self.subsample_grid_size = subsample_grid_size
        self.device = device
        self.num_surface_points = num_surface_points_per_obj
        self.conf = 0.5
        self.bhm = None

    def fit(self, scene: Scene) -> None:
        hinge_points = generate_hingepoint_grid_multiple_objects_w_surface(
            scene.points, scene.seg_mask, self.grid_len, self.grid_dist_from_obj, self.num_surface_points
        )
        print(f"hinge_points: {hinge_points.shape}")
        kernel = GaussianKernel(self.kernel_param)
        # (2) create bhm
        bhm = VPRISM(
            num_classes=int(torch.amax(scene.seg_mask).item()) + 1,
            hinge_points=hinge_points,
            kernel=kernel,
            num_iterations=3,
            # num_iterations_initial=10
            # num_iterations=5,
        ).to(self.device)
        # (3) neg sampling on data
        scene_center = 0.5 * (
            torch.amax(scene.points[scene.seg_mask > 0], dim=0) 
            + torch.amin(scene.points[scene.seg_mask > 0], dim=0)
        )
        X, y = negative_sample_each_ray_for_ablation(scene.points, scene.seg_mask, scene.camera_pos)
        mask = torch.norm(X - scene_center, dim=1) <= self.scene_sphere_radius
        X = X[mask]
        y = y[mask]
        print(f"X: {X.shape}")
        # (4) fit bhm
        bhm.sequential_update(X.to(self.device), y.to(self.device), 20000)
        self.bhm = bhm

    def predict(self, x: Tensor) -> Tensor:
        return self.bhm.predict(x.to(self.device)).to(torch.device("cpu"))


class SigmoidBHMsReconAlgorithm:
    def __init__(
        self, 
        grid_len: float, 
        grid_dist_from_obj: float, 
        kernel_param: float,
        ray_step_size: float,
        object_sphere_radius: float,
        scene_sphere_radius: float,
        subsample_grid_size_occ: float,
        subsample_grid_size_unocc: float,
        num_surface_points_per_obj: int,
        device: torch.device = device
    ) -> None:
        self.grid_len = grid_len
        self.grid_dist_from_obj = grid_dist_from_obj
        self.kernel_param = kernel_param
        self.ray_step_size = ray_step_size
        self.object_sphere_radius = object_sphere_radius
        self.scene_sphere_radius = scene_sphere_radius
        self.subsample_grid_size_occ = subsample_grid_size_occ
        self.subsample_grid_size_unocc = subsample_grid_size_unocc
        self.device = device
        self.num_surface_points = num_surface_points_per_obj
        self.conf = 0.5
        self.bhms = None

    def fit(self, scene: Scene) -> None:
        hinge_points = generate_hingepoint_grid_multiple_objects_w_surface(
            scene.points, scene.seg_mask, self.grid_len, self.grid_dist_from_obj, self.num_surface_points
        )
        print(f"hinge_points: {hinge_points.shape}")
        kernel = GaussianKernel(self.kernel_param)
        num_objects = int(torch.amax(scene.seg_mask).item())
        # (2) create bhm
        bhms = [(
            BayesianHilbertMapWithFullCovarianceMatrixNoInv(
                hinge_points=hinge_points,
                kernel=kernel,
                num_iterations=3,
                # num_iterations_initial=5,
            ).to(self.device)
        ) for i in range(num_objects)]
        # (3) neg sampling on data
        scene_center = 0.5 * (
            torch.amax(scene.points[scene.seg_mask > 0], dim=0) 
            + torch.amin(scene.points[scene.seg_mask > 0], dim=0)
        )
        X, y = negative_sample_rays_in_sphere_uniform_each_step_multiclass(
            points=scene.points, 
            mask=scene.seg_mask, 
            step_size=self.ray_step_size, 
            radius=self.object_sphere_radius, 
            camera_pos=scene.camera_pos,
        )
        plane = robust_ransac(scene.points, scene.seg_mask, 100, dist_tol=0.01, radius=self.scene_sphere_radius)
        X, y = add_negative_points_below_plane_multiple_objects(
            X, y, plane=[*list(plane.normal_vect), -plane.bias], center=scene_center, radius=self.scene_sphere_radius, k=10000
        )
    
        X, y = grid_subsample_different_res(
            X, 
            y, 
            subsample_grid_size_unocc=self.subsample_grid_size_unocc, 
            subsample_grid_size_occ=self.subsample_grid_size_occ
        )
        print(f"X: {X.shape}")
        # (4) fit bhm
        for i, bhm in enumerate(bhms):
            bhm.sequential_update(X.to(self.device), (y == i+1).to(self.device), 10000)
        self.bhms = bhms

    def predict(self, x: Tensor) -> Tensor:
        preds = torch.stack([bhm.predict(x.to(self.device)).to(torch.device("cpu")) for bhm in self.bhms], dim=1)  # (P, C-1)
        empty_pred = 1 - torch.sum(preds, dim=1, keepdim=True)  # (P, 1)
        preds = torch.concatenate([empty_pred, preds], dim=1)  # (P, C)
        return preds

class VoxelBaselineAlgorithm:
    def __init__(
        self, 
        grid_length: float,
        grid_dist_from_obj: float,
        scene_sphere_radius: float,
    ) -> None:
        self.grid_length = grid_length
        self.grid_dist_from_obj = grid_dist_from_obj
        self.grid = None
        self.scene_sphere_radius = scene_sphere_radius
        self.conf = 0.5

    def fit(self, scene: Scene) -> None:
        # (0) init grids + grid_subsample
        grid = generate_hingepoint_grid_multiple_objects_w_surface(
            scene.points, 
            scene.seg_mask, 
            resolution=self.grid_length,
            dist_from_obj=self.grid_dist_from_obj
        ).to(device)
        print(f"grid.shape {grid.shape}")
        P = grid.shape[0]
        labels = torch.zeros_like(grid[:, 0], dtype=torch.int)
        observed = torch.zeros_like(labels)
        scene = scene.to(device)
        points_subsampled, seg_subsampled = grid_subsample(scene.points, scene.seg_mask, subsample_grid_size=self.grid_length)
        self.num_classes = int(torch.amax(scene.seg_mask).item() + 1)
        object_centers_list = []
        for id in seg_subsampled[seg_subsampled > 0].unique():
            obj_points = points_subsampled[seg_subsampled == id]
            object_centers_list.append(0.5 * (torch.amax(obj_points, dim=0) + torch.amin(obj_points, dim=0)))
        obj_centers = torch.stack(object_centers_list)  # (N, 3)
        depth = torch.norm(points_subsampled - scene.camera_pos, dim=1)
        normalized_points = (points_subsampled - scene.camera_pos) / (depth.unsqueeze(1) + EPSILON)
        normalized_centers = (obj_centers - scene.camera_pos) / torch.norm(obj_centers - scene.camera_pos, dim=1, keepdim=True)
        closest_rays = torch.argmax(normalized_centers @ normalized_points.T, dim=0)  # (P,) of ints
        projections = torch.sum(normalized_centers[closest_rays] * (points_subsampled - scene.camera_pos), dim=1, keepdim=True) * normalized_centers[closest_rays]
        mask_near_center = torch.norm(projections - points_subsampled + scene.camera_pos, dim=1) <= self.grid_dist_from_obj
        points_subsampled = points_subsampled[mask_near_center] 
        seg_subsampled = seg_subsampled[mask_near_center] 
        print(f"points subsampled {points_subsampled.shape}")

        # (1) label voxels observed as 0 (store points)
        depth = torch.norm(points_subsampled - scene.camera_pos, dim=1)
        normalized_points = (points_subsampled - scene.camera_pos) / (depth.unsqueeze(1) + EPSILON)
        normalized_grid = (grid - scene.camera_pos) / torch.norm(grid - scene.camera_pos, dim=1, keepdim=True)
        closest_rays = torch.argmax(normalized_grid @ normalized_points.T, dim=1)  # (G,) of ints
        projections =  torch.sum(normalized_points[closest_rays] * (grid - scene.camera_pos), dim=1, keepdim=True) * normalized_points[closest_rays]
        mask_near_ray = torch.norm(projections - grid + scene.camera_pos, dim=1) <= self.grid_length
        mask_in_front_of = torch.norm(projections, dim=1) <= depth[closest_rays] - self.grid_length
        mask_observed = torch.logical_and(mask_near_ray, mask_in_front_of)
        points_empty = projections[mask_observed] + scene.camera_pos
        observed[mask_observed] = 1

        # (2) label voxels with ray termination as label
        dists = torch.cdist(points_subsampled.unsqueeze(0), grid.unsqueeze(0)).squeeze()  # (P, G)
        closest_idx = torch.argmin(dists, dim=1)  # (P,)
        min_dists = torch.amin(dists, dim=1)
        mask = min_dists < self.grid_length + EPSILON
        closest_idx_filtered = closest_idx[mask]  # <-- PROBLEM
        labels[closest_idx_filtered] = seg_subsampled[mask]
        observed[closest_idx_filtered] = 1

        # (3) label each unlabeled voxel w closest point stored there
        not_observed = observed == 0
        all_points = torch.concat([points_subsampled, points_empty], dim=0)
        all_labels = torch.concat([seg_subsampled, torch.zeros_like(points_empty[:, 0], dtype=torch.int)], dim=0)
        grid_not_filled = grid[not_observed]
        dists = torch.cdist(all_points.unsqueeze(0), grid_not_filled.unsqueeze(0)).squeeze()  # (P', G')
        min_dist_idx = torch.argmin(dists, dim=0)
        labels[not_observed] = all_labels[min_dist_idx]

        # (4) label voxels under table as unoccupied
        plane = robust_ransac(scene.points, scene.seg_mask, 100, dist_tol=0.01, radius=self.scene_sphere_radius)
        mask = grid @ plane.normal_vect + plane.bias < 0
        labels[mask] = 0

        self.grid = grid
        self.labels = labels

    def predict(self, x: Tensor) -> Tensor:
        dists = torch.cdist(self.grid.unsqueeze(0), x.to(device).unsqueeze(0)).squeeze()
        idxs = torch.argmin(dists, dim=0)
        return torch.eye(self.num_classes, device=device)[self.labels[idxs]].to(torch.device("cpu"))  # (P, C)


# this probably looks confusing... but it works.
def get_func_wrapper(view_mat):
    def wrapper(func):
        def new_func(x):
            x_aug = torch.concat([x, torch.ones_like(x[:, :1])], dim=1)
            out = x_aug @ view_mat.T
            return func(out[:, :3])
        return new_func
    return wrapper


model_dir = abspath(os.path.join(logs_dir, "models"))
class NeuralNetworkAlgorithm:
    def __init__(self, model_name: str, npoints: int = 256) -> None:
        model_path = os.path.join(model_dir, model_name)
        self.model: PointSDF = torch.load(model_path)
        self.model.to(device)
        self.conf = 0.3
        self.npoints = npoints

    def fit(self, scene: Scene) -> None:
        obj_points = scene.points[scene.seg_mask > 0]
        scene_points = scene.points.to(device)  # for speed up!
        object_points_list = []
        for i, metadata in enumerate(scene.object_metadata):
            point_cloud = scene_points[scene.seg_mask == i+1]
            sampled_points_idx = farthest_point_sample(point_cloud.unsqueeze(0), npoint=self.npoints)
            sampled_points = index_points(point_cloud.unsqueeze(0), sampled_points_idx)
            object_points_list.append(sampled_points.reshape(self.npoints, 3))
        obj_points_uncentered = torch.stack(object_points_list).to(torch.float).cpu().to(torch.float)
        obj_points, centers = scale_and_center_object_points(obj_points_uncentered)
        self.N = centers.shape[0]
        self.model.to(device)
        self.model.eval()
        self.centers = centers
        with torch.no_grad():
            self.obj_feats = self.model.get_latent_features(obj_points.to(device))

    def predict(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            query_pts = scale_and_center_queries(self.centers.to(device), x.to(torch.float).to(device).unsqueeze(0).repeat((self.N, 1, 1)))
            preds = self.model.get_preds(self.obj_feats, query_pts)  # (N, P)
        preds = preds.T  # (P, N)
        empty_pred = 1 - torch.sum(preds, dim=1, keepdim=True)  # (P, 1)
        preds = torch.concatenate([empty_pred, preds], dim=1)
        return preds.cpu()
    


# (3) load stuff
if args.algorithm == "bhm":
    algo = SoftmaxBHMReconAlgorithm(
        grid_len=0.05,
        grid_dist_from_obj=0.15,
        kernel_param=1000,
        ray_step_size=0.1,
        object_sphere_radius=0.25,
        subsample_grid_size_occ=0.01,
        subsample_grid_size_unocc=0.015,
        num_surface_points_per_obj=32,
        scene_sphere_radius=0.4,
    )
elif args.algorithm == "no_stratified":
    algo = SoftmaxBHMReconNoStratifiedAlgorithm(
        grid_len=0.05,
        grid_dist_from_obj=0.15,
        kernel_param=1000,
        ray_step_size=0.1,
        object_sphere_radius=0.25,
        subsample_grid_size_occ=0.01,
        subsample_grid_size_unocc=0.015,
        num_surface_points_per_obj=32,
        scene_sphere_radius=0.4,
    )
elif args.algorithm == "no_under_table":
    algo = SoftmaxBHMReconNoUnderTableAlgorithm(
        grid_len=0.05,
        grid_dist_from_obj=0.15,
        kernel_param=1000,
        ray_step_size=0.1,
        object_sphere_radius=0.25,
        subsample_grid_size_occ=0.01,
        subsample_grid_size_unocc=0.015,
        num_surface_points_per_obj=32,
        scene_sphere_radius=0.4,
    )
elif args.algorithm == "bad_sampling":
    algo = SoftmaxBHMReconAlgorithmBadSampling(
        grid_len=0.05,
        grid_dist_from_obj=0.15,
        kernel_param=1000,
        ray_step_size=0.1,
        object_sphere_radius=0.25,
        subsample_grid_size=0.015,
        num_surface_points_per_obj=32,
        scene_sphere_radius=0.4,
    )
elif args.algorithm == "sigmoid":
    algo = SigmoidBHMsReconAlgorithm(
        grid_len=0.05,
        grid_dist_from_obj=0.15,
        kernel_param=1000,
        ray_step_size=0.1,
        object_sphere_radius=0.25,
        subsample_grid_size_occ=0.01,
        subsample_grid_size_unocc=0.015,
        num_surface_points_per_obj=32,
        scene_sphere_radius=0.4,
    )
elif args.algorithm == "voxel":
    algo = VoxelBaselineAlgorithm(0.02, 0.2, 0.4)
elif args.algorithm == "neural_network":
    algo = NeuralNetworkAlgorithm("pointsdf.pt", 256)
else:
    raise Exception(f"there is no such algorithm '{args.algorithm}'")

dataset = ReconBulletDataset(args.dataset, data_dir=data_dir, keep_camera_at_origin=args.camera_at_origin)

logfile = os.path.join(logs_dir, args.output)

# (4) loop through data
print_and_maybe_log("================================", filename=logfile)
print_and_maybe_log(f"Evaluating {args.algorithm} algo on {args.dataset} dataset.", filename=logfile)
ious = []
soft_ious = []
chamfers = []
for i in range(len(dataset) // dataset.num_views_per):  # only first view of each scene
    scene = dataset[i * dataset.num_views_per]
    start = time.time()
    algo.fit(scene)
    end_fit = time.time()
    print(f"scene {i} fit in {end_fit - start} seconds.")
    if args.camera_at_origin:
        view_dir = os.path.join(data_dir, args.dataset, f"{i:08d}", "0000")
        camera_data = np.load(os.path.join(view_dir, "camera.npz"))
        view_mat = from_numpy(camera_data["view_matrix"])
        func_wrapper = get_func_wrapper(view_mat)
    for i, obj_data in enumerate(scene.object_metadata):
        mesh_path = os.path.join(data_dir, obj_data["mesh_path"])
        func = lambda x: algo.predict(x)[:, i+1]
        if args.camera_at_origin:
            func = func_wrapper(func)
        iou = calc_iou(
            mesh_path,
            pred_func=func,
            resolution=RESOLUTION,
            mesh_position=np.array(obj_data["position"]),
            mesh_orientation=np.array(obj_data["orientation"]),
            mesh_scale=np.array(obj_data["scale"]),
            conf=algo.conf,
        )
        chamfer = chamfer_dist_for_mesh(
            mesh_path,
            pred_func=func,
            resolution=RESOLUTION,
            mesh_position=np.array(obj_data["position"]),
            mesh_orientation=np.array(obj_data["orientation"]),
            mesh_scale=np.array(obj_data["scale"]),
            conf=algo.conf,
        )
        ious.append(iou)
        if chamfer is not torch.nan:
            chamfers.append(chamfer)
        print(f"metrics: {iou} {chamfer}")
    end_calcs = time.time()
    print(f"metrics calculated in {end_calcs - end_fit} seconds")
    if len(chamfers) == 0:
        continue
    print_and_maybe_log(f"running avg metrics: {sum(ious) / len(ious)} {sum(chamfers) / len(chamfers)}", filename=logfile)


# (5) compute summary + print
print_and_maybe_log("--------------------------------", filename=logfile)

print_and_maybe_log(f"Iou of {sum(ious) / len(ious)}", filename=logfile)
print_and_maybe_log(f"Chamfer of {sum(chamfers) / len(chamfers)}", filename=logfile)
print_and_maybe_log("================================", filename=logfile)


