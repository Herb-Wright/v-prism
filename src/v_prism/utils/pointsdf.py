from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import Module, Linear, Sequential, ReLU, Dropout, BatchNorm1d, Sigmoid, Tanh
from torch.nn import Identity, functional as F

SCALE = 0.15  # gets mapped s.t. [-0.15, 0.15] -> [-1, 1]

def scale_and_center_object_points(points: Tensor) -> tuple[Tensor, Tensor]:
    centers = 0.5 * (torch.amax(points, dim=1) + torch.amin(points, dim=1))  # (N, 3)
    centered_points = points - centers.unsqueeze(1)
    scaled_points = centered_points / SCALE
    return scaled_points, centers

def scale_and_center_queries(centers: Tensor, query_pts: Tensor) -> Tensor:
    query_pts_centered = query_pts - centers.unsqueeze(1)
    query_pts_scaled = query_pts_centered / SCALE
    return query_pts_scaled


class PointSDF(Module):
    def __init__(
        self, 
        query_encoder_dropout = 0.2, 
        final_activation: str = "sigmoid",
    ):
        super().__init__()
        self.pointconv_encoder = PointConvEncoder()
        self.query_encoder = Sequential(
            Linear(3, 512),
            ReLU(),
            Dropout(query_encoder_dropout),
            Linear(512, 256),
            ReLU(),
            Dropout(query_encoder_dropout)
        )
        self.sdf_model1 = Sequential(
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 256),
            BatchNorm1d(256),
            ReLU(),
        )
        self.sdf_model2 = Sequential(
            Linear(512+256, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 512),
            BatchNorm1d(512),
            ReLU(),
            Linear(512, 1),
        )
        if final_activation == "sigmoid":
            self.final_activation = Sigmoid()
        elif final_activation == "tanh":
            self.final_activation = Tanh()
        elif final_activation == "none":
            self.final_activation = Identity()
        else:
            raise Exception(f"unknown activation: '{final_activation}'")
        

    def get_latent_features(self, points: Tensor) -> Tensor:
        feats = self.pointconv_encoder(points)
        return feats

    def get_preds(self, latent_feats: Tensor, query_pts: Tensor) -> Tensor:
        Q = query_pts.shape[1]
        N = latent_feats.shape[0]
        query_feats = self.query_encoder(query_pts)
        latent_feats_expanded = latent_feats.unsqueeze(1).repeat((1, Q, 1))  # (N, Q, H)
        cat_feats = torch.concat([latent_feats_expanded, query_feats], dim=2).reshape(N * Q, 512)
        out = self.sdf_model1(cat_feats)
        cat_out = torch.cat((out, cat_feats), 1)  # (N * Q, H + H_1)
        out = self.sdf_model2(cat_out)
        return self.final_activation(out).reshape(N, Q)

    def forward(self, points: Tensor, query_pts: Tensor) -> Tensor:
        feats = self.get_latent_features(points)
        preds = self.get_preds(feats, query_pts)
        return preds



class PointConvEncoder(Module):
    def __init__(self) -> None:
        super().__init__()
        feature_dim = 3
        # NOTE: the following arguments correspond to arguments in the tensorflow version's feature_encoding_layer:
        #   - npoint -> npoint
        #   - nsample -> K
        #   - mlp -> mlp
        #   - bandwith -> sigma 
        # other arguments have no direct correspondence
        self.sa1 = PointConv(npoint=512, nsample=32, in_channel=feature_dim+3, mlp=[32, 32, 64], bandwidth=0.05, group_all=False)
        self.sa2 = PointConv(npoint=256, nsample=32, in_channel=64+3, mlp=[64, 64, 64], bandwidth=0.1, group_all=False)
        self.sa3 = PointConv(npoint=64, nsample=32, in_channel=64+3, mlp=[128, 128, 256], bandwidth=0.2, group_all=False)
        self.sa4 = PointConv(npoint=36, nsample=32, in_channel=256+3, mlp=[256, 256, 512], bandwidth=0.4, group_all=False)
        self.fc1 = Linear(36*512,256)
        self.bn1 = BatchNorm1d(256)

    def forward(self, points: Tensor) -> Tensor:
        feat = points
        points = points.transpose(2,1)
        l1_xyz, l1_points = self.sa1(points, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        _, l4_points = self.sa4(l3_xyz, l3_points)
        embedding = torch.reshape(l4_points, (-1, 36*512))
        cloud_embedding = self.fc1(embedding)
        if cloud_embedding.shape[0] > 1:
            cloud_embedding = self.bn1(cloud_embedding)
        cloud_embedding = F.relu(cloud_embedding)
        return cloud_embedding


# =============================================================================================================
#    THIS IS COPIED FROM https://github.com/DylanWusee/pointconv_pytorch/blob/master/utils/pointconv_util.py

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    #import ipdb; ipdb.set_trace()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N, dtype=xyz.dtype).to(device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long).to(device)
    # farthest = torch.randint(N, size=(B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.argmax(distance, -1)
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).broadcast_to([B, S, N])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def sample_and_group(npoint, nsample, xyz, points, density_scale = None):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if density_scale is None:
        return new_xyz, new_points, grouped_xyz_norm, idx
    else:
        grouped_density = index_points(density_scale, idx)
        return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

def sample_and_group_all(xyz, points, density_scale = None):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    #new_xyz = torch.zeros(B, 1, C).to(device)
    new_xyz = xyz.mean(dim = 1, keepdim = True)
    grouped_xyz = xyz.view(B, 1, N, C) - new_xyz.view(B, 1, 1, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    if density_scale is None:
        return new_xyz, new_points, grouped_xyz
    else:
        grouped_density = density_scale.view(B, 1, N, 1)
        return new_xyz, new_points, grouped_xyz, grouped_density

def group(nsample, xyz, points):
    """
    Input:
        npoint:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm

def compute_density(xyz, bandwidth):
    """
    xyz: input points position data, [B, N, C]

    returns: (b, n)
    """
    #import ipdb; ipdb.set_trace()
    B, N, C = xyz.shape
    # sqrdists = square_distance(xyz, xyz)
    # print(sqrdists.shape)
    # print(bandwidth)
    # gaussion_density = - sqrdists
    # gaussion_density = gaussion_density / (2.0 * bandwidth ** 2)
    # gaussion_density = torch.exp(gaussion_density)
    # gaussion_density = gaussion_density / (2.5 * bandwidth)
    # gaussion_density = torch.exp(- sqrdists / (2.0 * bandwidth ** 2)) / (2.5 * bandwidth)
    out = - square_distance(xyz, xyz) * 2.0 * bandwidth ** 2  # (b, n, n)
    out.exp_()  # (b, n, n)
    out /= (2.5 * bandwidth)  # (b, n, n)
    out = out.mean(dim=-1)  # (b, n)
    return out


class DensityNet(nn.Module):
    def __init__(self, hidden_unit = [16, 8]):
        super(DensityNet, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList() 

        self.mlp_convs.append(nn.Conv2d(1, hidden_unit[0], 1))
        self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
        for i in range(1, len(hidden_unit)):
            self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
        self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], 1, 1))
        self.mlp_bns.append(nn.BatchNorm2d(1))

    def forward(self, density_scale):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            density_scale =  bn(conv(density_scale))
            if i == len(self.mlp_convs):
                density_scale = F.sigmoid(density_scale)
            else:
                density_scale = F.relu(density_scale)
        
        return density_scale

class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights

class PointConv(nn.Module):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all):
        super(PointConv, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet = WeightNet(3, 16)
        self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
        self.bn_linear = nn.BatchNorm1d(mlp[-1])
        self.densitynet = DensityNet()
        self.group_all = group_all
        self.bandwidth = bandwidth

    def forward(self, xyz: Tensor, points: Tensor) -> tuple[Tensor, Tensor]:
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        # make sure channels are in the right order
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)  # (b, n, c); usually c=3
        if points is not None:
            points = points.permute(0, 2, 1)

        # compute density
        xyz_density = compute_density(xyz, self.bandwidth)
        inverse_density = 1.0 / xyz_density 

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = sample_and_group_all(xyz, points, inverse_density.view(B, N, 1))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = sample_and_group(self.npoint, self.nsample, xyz, points, inverse_density.view(B, N, 1))
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        inverse_max_density = grouped_density.max(dim = 2, keepdim=True)[0]
        density_scale = grouped_density / inverse_max_density
        density_scale = self.densitynet(density_scale.permute(0, 3, 2, 1))
        new_points = new_points * density_scale

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        C1 = weights.shape[1]
        C2 = new_points.shape[1]
        new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).reshape(B, self.npoint, C1 * C2)
        new_points = self.linear(new_points)
        new_points = self.bn_linear(new_points.permute(0, 2, 1))
        new_points = F.relu(new_points)
        new_xyz = new_xyz.permute(0, 2, 1)

        return new_xyz, new_points

