from dataclasses import dataclass

from torch import Tensor, device


@dataclass
class Scene:
    points: Tensor  # (P, D)
    camera_pos: Tensor   # (D,)
    seg_mask: Tensor | None = None  # (P,)
    obj_ids: list[int] | None = None
    sdf_vals: Tensor | None = None  # (N, Q)
    query_pts: Tensor | None = None  # (N, Q, D) generally D = 3
    object_metadata: list[dict[str, object]] | None = None

    def __str__(self) -> str:
        segmask = self.seg_mask.shape if self.seg_mask is not None else 'None'
        return f'Scene(points={self.points.shape}, seg_mask={segmask}, camera_pos={self.camera_pos})'

    def to(self, device: device):
        return Scene(
            self.points.to(device),
            self.camera_pos.to(device),
            self.seg_mask.to(device) if self.seg_mask is not None else None,
            self.obj_ids,
            self.sdf_vals.to(device) if self.sdf_vals is not None else None,
            self.query_pts.to(device) if self.query_pts is not None else None,
            self.object_metadata
        )


