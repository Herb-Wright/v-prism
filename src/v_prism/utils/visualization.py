import io
from typing import Callable
from logging import info, warn

from trimesh import Trimesh, PointCloud, Scene
from PIL import Image
from PIL.Image import Image as PILImage
from numpy.typing import NDArray
import torch
from torch import Tensor
from tqdm import trange
from skimage import measure
import numpy as np


_RESOLUTION = 0.01

def gen_mesh_for_sdf_batch_3d_numpy(
    occ_func: Callable[[NDArray], NDArray], 
    xlim: list[int] = [0, 1], 
    ylim: list[int] = [0, 1], 
    zlim: list[int] = [0, 1],
    resolution: float = _RESOLUTION,
    *,
    confidence: float = 0.5
) -> Trimesh:
    """Performs marching cubes to generate a mesh for a given sdf 
    
    (done in batch so it is faster)

    Args:
        - occ_func: (Callable: (N, 3) -> (N,))
        - xlim: (list[int]) list of two values corresponding to max and min x
        - ylim: (list[int]) list of two values corresponding to max and min y
        - zlim: (list[int]) list of two values corresponding to max and min z
        - resolution: (float) the resolution for the mesh
    
    Returns:
        - mesh: (Trimesh)
    """
    xyz = np.stack(np.meshgrid(
        np.arange(xlim[0], xlim[1], step=resolution),
        np.arange(ylim[0], ylim[1], step=resolution),
        np.arange(zlim[0], zlim[1], step=resolution),
    ), axis=-1)
    X, Y, Z, _ = xyz.shape
    xyz = xyz.reshape((xyz.shape[0], -1, 3))
    tsdf = np.zeros((X, Y * Z))
    for i in trange(X):
        tsdf[i] = occ_func(xyz[i]) - confidence
    tsdf = tsdf.reshape(X, Y, Z)
    info(f'reconstructing mesh; max={tsdf.max()}, min={tsdf.min()}')
    verts, faces, normals, _ = measure.marching_cubes(
        tsdf, 
        0, 
        spacing=[resolution for i in range(3)],
    )
    idx = np.array([1, 0, 2], dtype=np.int64)
    verts = verts[:, idx]
    normals = normals[:, idx]
    verts = verts + np.array([xlim[0], ylim[0], zlim[0]])
    mesh = Trimesh(verts, faces, normals)
    mesh.visual.vertex_colors = np.array([100, 100, 200, 215])
    return mesh

def gen_mesh_for_sdf_batch_3d(
    occ_func: Callable[[NDArray], NDArray], 
    xlim: list[int] = [0, 1], 
    ylim: list[int] = [0, 1], 
    zlim: list[int] = [0, 1],
    resolution: float = _RESOLUTION,
    *,
    confidence: float = 0.5,
    device: torch.device = torch.device('cpu'),
    only_batch_one_dim: bool = False,
) -> Trimesh:
    """Performs marching cubes to generate a mesh for a given sdf 
    
    (done in batch so it is faster)

    Args:
        - occ_func: (Callable: (N, 3) -> (N,))
        - xlim: (list[int]) list of two values corresponding to max and min x
        - ylim: (list[int]) list of two values corresponding to max and min y
        - zlim: (list[int]) list of two values corresponding to max and min z
        - resolution: (float) the resolution for the mesh
        - (optional) confidence: (float) the level set value. Defaults to 0.5
        - (optional) device: (torch.device) device for calculation. Defaults to cpu
        - (optional) only_batch_one_dim: (bool) flag for performance. Don't touch, defaults to false.
    
    Returns:
        - mesh: (Trimesh)
    """
    xyz = torch.tensor(np.stack(np.meshgrid(
        np.arange(xlim[0], xlim[1], step=resolution),
        np.arange(ylim[0], ylim[1], step=resolution),
        np.arange(zlim[0], zlim[1], step=resolution),
    ), axis=-1), device=device)
    X, Y, Z, _ = xyz.shape
    if only_batch_one_dim:
        tsdf = torch.zeros((X, Y, Z), device=device)
        for i in trange(X):
            for j in range(Y):
                # print('here', xyz[i, j].shape)
                tsdf[i, j] = occ_func(xyz[i, j])
    else:
        xyz = xyz.reshape((xyz.shape[0], -1, 3))
        tsdf = torch.zeros((X, Y * Z), device=device)
        for i in trange(X):
            tsdf[i] = occ_func(xyz[i]) - confidence
        tsdf = tsdf.reshape(X, Y, Z)
    info(f'reconstructing mesh; max={tsdf.amax()}, min={tsdf.amin()}')
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            tsdf.cpu().numpy(), 
            0, 
            spacing=[resolution for i in range(3)],
        )
        idx = np.array([1, 0, 2], dtype=np.int64)
        verts = verts[:, idx]
        normals = normals[:, idx]
        verts = verts + np.array([xlim[0], ylim[0], zlim[0]])
        mesh = Trimesh(verts, faces, normals)
        mesh.visual.vertex_colors = np.array([100, 100, 200, 215])
    except Exception as err:
        warn(err)
        warn('error while running marching cubes; returning empty mesh')
        mesh = None
    return mesh


def gen_pointcloud_for_X_y(X: NDArray | Tensor, y: NDArray | Tensor) -> PointCloud:
    """generates a colored trimesh pointcloud for the (X, y) pair 

    uses red for empty, blue for occupied.

    Args:
        - X: (P, 3) array of points
        - y: (P,) binary vector

    Returns:
        - pointcloud: (PointCloud)
    """
    if isinstance(X, Tensor):
        X.cpu().numpy()
    if isinstance(y, Tensor):
        y.cpu().numpy()
    colors = np.array([
        [255, 0, 0],
        [0, 0, 255]
    ])
    colors_for_pointcloud = colors[y.to(torch.long)]
    return PointCloud(vertices=X, colors=colors_for_pointcloud)



def gen_image_of_trimesh_scene(
    scene: Scene, 
    theta: float,
    *,
    phi: float = 0.35 * np.pi,
    rho: float = 0.35,
    lookat_position: NDArray | None = None,
    rotate: bool = True,
    pi: float = 0.0,
    resolution: list[float] = (512, 512),
    line_settings: dict | None = None
) -> PILImage:
    """returns a color image of the trimesh scene

    Args:
        - scene: (trimesh.Scene) trimesh scene to take img of
        - theta: (float) angle around ground plane in radians
        - phi: (float) angle from up direction in radians. default to 0.35*pi.
        - rho: (float) distance from center of scene. default to 0.35.
        - lookat_position: (3,) position in middle of screen
    
    Returns:
        - img: (H, W, 3) image of scene
    """
    if lookat_position is None:
        lookat_position = np.mean(scene.bounds, axis=0)
    scene.set_camera(
        np.array([pi, phi, theta]),
        distance=rho,
        center=lookat_position,
    )
    img = Image.open(io.BytesIO(scene.save_image(resolution=resolution, line_settings=line_settings)))
    if rotate:
        img = img.rotate(-90)
    return img


def gen_heatmap(
    func: Callable[[Tensor], Tensor],
    limits: Tensor,
    step: float = 0.002,
    *,
    z: float | None,
) -> Tensor:
    """generates heatmap for func
    
    Args:
        - func: ((P, 3) -> (P,))
        - limits: (2, 3) contains the min values in limits[0] and max in limits[1]
        - step: (float) grid step to do resolution at.
        - [optional] z: (float | None) the z level. If `None`, defaults to  0.5 * (limits[0, 2] + limits[1, 2])

    Returns:
        - heatmap image
    """
    points = torch.tensor(np.stack(np.meshgrid(
        np.arange(limits[0, 0], limits[1, 0], step=step),
        np.arange(limits[0, 1], limits[1, 1], step=step),
        torch.tensor(z),
    ), axis=-1), device=limits.device)
    H, W, _, D = points.shape
    out = func(points.reshape((H * W, D)))
    return out.reshape(H, W)


