
import os
import json
from logging import debug

from PIL import Image
import numpy as np
from numpy.typing import NDArray
import torch

from ..settings import DATA_DIR
from ..utils.data_download import download_file_from_google_drive, unzip_tar_file
from .scene import Scene


# # https://drive.google.com/file/d/1HDJYidwxm0CiIHUvjyzCIJsC5jp1VNj2/view?usp=drive_link
# def download_simple_ycb_dataset(data_dir = DATA_DIR) -> None:
#     tar_file_path = os.path.join(data_dir, "gsdf_bullet_ycb_data.tar.xz")
#     file_path = os.path.join(data_dir, "gsdf_bullet_ycb_train")
#     if not os.path.exists(file_path):
#         try:
#             if not os.path.exists(tar_file_path):
#                 print("Downloading gsdf_bullet_ycb_data dataset from google drive.")
#                 download_file_from_google_drive("1HDJYidwxm0CiIHUvjyzCIJsC5jp1VNj2", tar_file_path, verbose=True)
#             print("Extracting downloading gsdf_bullet_ycb_data .tar.xz file")
#             unzip_tar_file(tar_file_path, data_dir, verbose=True, num=48804)
#         except Exception as e:
#             print(f"Unable to download dataset ({e})")
#             print(e.with_traceback())


class ReconBulletDataset:
    def __init__(
        self,
        dataset_name: str,
        data_dir: str = DATA_DIR,
        *,
        keep_camera_at_origin: bool = False
    ) -> None:
        self.keep_camera_at_origin = keep_camera_at_origin
        self.dataset_dir = os.path.join(data_dir, dataset_name)
        self.datafiles = [(
            os.path.join(self.dataset_dir, scene)
        ) for scene in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, scene))]
        self.datafiles = sorted(self.datafiles)
        config_path = os.path.join(self.dataset_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
                self.num_views_per = config["num_views_per_scene"]
        else:
            self.num_views_per = len([d for d in os.listdir(self.datafiles[0]) if os.path.isdir(os.path.join(self.datafiles[0], d))])


    def __len__(self) -> int:
        return len(self.datafiles) * self.num_views_per

    def __getitem__(self, idx: int) -> Scene:
        scene_idx = idx // self.num_views_per
        view_idx = idx % self.num_views_per
        return self.get_scene_view(scene_idx, view_idx, use_view_matrix=(not self.keep_camera_at_origin))

    def get_scene_view(
        self, 
        scene_idx: int, 
        view_idx: int, 
        *, 
        use_view_matrix: bool = True
    ) -> Scene:    
        scene_dir = self.datafiles[scene_idx]
        view_dirs = [(
            os.path.join(scene_dir, view)
        ) for view in sorted(os.listdir(scene_dir)) if os.path.isdir(os.path.join(scene_dir, view))]
        view_dir = view_dirs[view_idx]
        debug(f"retrieving view from dir '{view_dir}'")
        rgb = np.array(Image.open(os.path.join(view_dir, "rgb.jpg")), dtype=float) / 256  # (3, H, W)
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.load(os.path.join(view_dir, "depth.npy"))  # (H, W)
        depth[depth < 1e-8] = 1e3  # make 0 depth really far.
        seg_mask: NDArray = np.load(os.path.join(view_dir, "seg_mask.npy"))  # (H, W)
        camera_data = np.load(os.path.join(view_dir, "camera.npz"))
        view_matrix = camera_data["view_matrix"]
        xyz = self._depth_to_xyz(
            depth, 
            camera_data["projection_matrix"], 
            view_matrix if use_view_matrix else np.eye(4)
        )
        with open(os.path.join(scene_dir, "objects.json"), "r") as f:
            objects_data = json.load(f)
        ids = []
        idxs = []
        metadata = []
        unique_ids = np.unique(seg_mask)
        for i, obj in enumerate(objects_data):
            if obj["id"] in unique_ids and np.sum(seg_mask == obj["id"]) >= 16:  # we want there to be at least 16 points
                idxs.append(i)
                ids.append(len(idxs))
                seg_mask[seg_mask == obj["id"]] = len(idxs)
                metadata.append(objects_data[i])
        
        idxs_np = np.array(idxs)
        if os.path.exists(os.path.join(scene_dir, "query_pts.npy")):
            query_pts = np.load(os.path.join(scene_dir, "query_pts.npy"))[idxs_np]
            if not use_view_matrix:
                query_pts_aug = np.concatenate((query_pts, np.ones_like(query_pts[:, :, 0:1])), axis=-1)
                query_pts = (query_pts_aug @ view_matrix.T)[:, :, :3]
            sdf_vals = np.load(os.path.join(scene_dir, "sdf_vals.npy"))[idxs_np]
        else:
            query_pts = None
            sdf_vals = None
        camera_pos = np.linalg.inv(view_matrix)[:3, 3] if use_view_matrix else np.zeros(3)
        return Scene(
            points=torch.tensor(xyz.reshape((-1, 3))),
            camera_pos=torch.tensor(camera_pos),
            seg_mask=torch.tensor(seg_mask.reshape((-1))),
            obj_ids=ids,
            sdf_vals=torch.tensor(sdf_vals) if sdf_vals is not None else None,
            query_pts=torch.tensor(query_pts) if query_pts is not None else None,
            object_metadata=metadata
        )


    # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
    def _depth_to_xyz(
        self, 
        depth: NDArray, 
        proj_matrix: NDArray, 
        view_matrix: NDArray = np.eye(4)
    ) -> NDArray:
        H, W = depth.shape
        tran_pix_world = np.linalg.inv(proj_matrix @ view_matrix)
        y, x = np.mgrid[-1:1:2 / H, -1:1:2 / W]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)
        pixels = np.stack([x, y, z, h], axis=1)
        pixels[:, 2] = 2 * pixels[:, 2] - 1
        points = (tran_pix_world @ pixels.T).T
        points = points / points[:, 3: 4]
        points: NDArray = points[:, :3]
        return points.reshape((H, W, 3))





