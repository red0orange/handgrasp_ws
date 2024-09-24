import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from .pointnet_util import PointNetSetAbstractionMsg, PointNetSetAbstraction, PointNetFeaturePropagation


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal embedding for time step.
    """
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        time = time * self.scale
        device = time.device
        half_dim = self.dim // 2

        # @note bug
        embeddings = math.log(10000) / (half_dim - 1) if half_dim > 1 else 0
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        embeddings = time.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def __len__(self):
        return self.dim
    

class PointNetPlusPlus(nn.Module):
    """_summary_
    PointNet++ class.
    """
    def __init__(self):
        super(PointNetPlusPlus, self).__init__()
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [
                                             32, 64, 128], 3, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(
            128, [0.4, 0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(
            npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)

        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=134, mlp=[128, 128])
        
        self.conv1 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)

    def forward(self, xyz):
        """_summary_
        Return point-wise features and point cloud representation.
        """
        # Set Abstraction layers
        xyz = xyz.contiguous().transpose(1, 2)
        l0_xyz = xyz
        l0_points = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        c = l3_points.squeeze()

        return None, c
        
        # # Feature Propagation layers
        # l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        # l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        # l0_points = self.fp1(l0_xyz, l1_xyz, torch.cat(
        #     [l0_xyz, l0_points], 1), l1_points)
        # l0_points = self.bn1(self.conv1(l0_points))
        # return l0_points, c


class UNetPoseNet(nn.Module):
    def __init__(self, action_dim, pointnet_dim=1024):
        super().__init__()
        point_feature_dim = pointnet_dim
        hidden_feature_dim = 256

        self.cloud_net3 = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_feature_dim),
            nn.GroupNorm(8, hidden_feature_dim),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, hidden_feature_dim)
        )
        self.cloud_net2 = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_feature_dim // 2),
            nn.GroupNorm(4, hidden_feature_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_feature_dim // 2, hidden_feature_dim // 2)
        )
        self.cloud_net1 = nn.Sequential(
            nn.Linear(point_feature_dim, hidden_feature_dim // 4),
            nn.GroupNorm(4, hidden_feature_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_feature_dim // 4, hidden_feature_dim // 4)
        )
        
        self.time_net3 = SinusoidalPositionEmbeddings(dim=hidden_feature_dim // 1)
        self.time_net2 = SinusoidalPositionEmbeddings(dim=hidden_feature_dim // 2)
        self.time_net1 = SinusoidalPositionEmbeddings(dim=hidden_feature_dim // 4)
        
        self.first_layer = nn.Sequential(
            nn.Linear(action_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, hidden_feature_dim)
        )
        self.down1 = nn.Sequential(
            nn.Linear(hidden_feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, hidden_feature_dim)
        )
        self.down2 = nn.Sequential(
            nn.Linear(hidden_feature_dim, hidden_feature_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_feature_dim // 2, hidden_feature_dim // 2)
        )
        self.down3 = nn.Sequential(
            nn.Linear(hidden_feature_dim // 2, hidden_feature_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_feature_dim // 4, hidden_feature_dim // 4)
        )
        
        self.up1 = nn.Sequential(
            nn.Linear(hidden_feature_dim // 4 + hidden_feature_dim // 2, hidden_feature_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_feature_dim // 2, hidden_feature_dim // 2)
        )
        self.up2 = nn.Sequential(
            nn.Linear(hidden_feature_dim // 2 + hidden_feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, hidden_feature_dim)
        )
        self.up3 = nn.Sequential(
            nn.Linear(hidden_feature_dim + hidden_feature_dim, hidden_feature_dim),
            nn.GELU(),
            nn.Linear(hidden_feature_dim, hidden_feature_dim)
        )
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_feature_dim, action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim)
        )
        pass
        
    def forward(self, g, c, context_mask, _t):
        """_summary_

        Args:
            g: pose representations, size [B, 7]
            c: point cloud representations, size [B, 1024]
            context_mask: masks {0, 1} for the contexts, size [B, 1]
            _t is for the timesteps, size [B,]
        """
        c = c * context_mask           # [B, 1024] -> [B, 1024]

        # Changed CNN
        c1 = self.cloud_net1(c)       # [B, 1024] -> [B, 128]
        c2 = self.cloud_net2(c)       # [B, 1024] -> [B, 256]
        c3 = self.cloud_net3(c)       # [B, 1024] -> [B, 512]
        
        _t0 = _t.unsqueeze(1)          # [B,] -> [B, 1]
        _t1 = self.time_net1(_t0)      # [B, 1] -> [B, 1, 128]
        _t2 = self.time_net2(_t0)      # [B, 1] -> [B, 1, 256]
        _t3 = self.time_net3(_t0)      # [B, 1] -> [B, 1, 512]
        _t1, _t2, _t3 = _t1.squeeze(), _t2.squeeze(), _t3.squeeze()  # [B, 1, 128] -> [B, 128], [B, 1, 256] -> [B, 256], [B, 1, 512] -> [B, 512]
        
        # @note 网络核心的 forward
        g = self.first_layer(g)        # [B, 7] -> [B, 512]
        g_down1 = self.down1(g)        # [B, 512] -> [B, 512]
        g_down2 = self.down2(g_down1)  # [B, 512] -> [B, 256]
        g_down3 = self.down3(g_down2)  # [B, 256] -> [B, 128]
        
        up1 = self.up1(torch.cat((g_down3 * c1 + _t1, g_down2), dim=1))   # [B, 2 + 4] -> [B, 4]
        up2 = self.up2(torch.cat((up1 * c2 + _t2, g_down1), dim=1))       # [B, 4 + 6] -> [B, 6]
        up3 = self.up3(torch.cat((up2 * c3 + _t3, g), dim=1))             # [B, 6 + 7] -> [B, 7]
        g = self.final_layer(up3)      # [B, 512] -> [B, 7]
        
        return g


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class TransformerPoseNet(nn.Module):
    def __init__(self, action_dim, pointnet_dim=1024):
        super().__init__()
        depth       = 8
        hidden_size = 256
        mlp_ratio   = 4.0
        num_heads   = 8
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.point_net = nn.Sequential(
            nn.Linear(pointnet_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.time_net = SinusoidalPositionEmbeddings(dim=hidden_size)
        self.up_net = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.GELU(),
            nn.Linear(128, hidden_size)
        )
        self.down_net = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, action_dim)
        )

        self.initialize_weights()
        pass

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        pass
        
    def forward(self, g, c, context_mask, _t):
        """_summary_

        Args:
            g: pose representations, size [B, 7]
            c: point cloud representations, size [B, 1024]
            context_mask: masks {0, 1} for the contexts, size [B, 1]
            _t is for the timesteps, size [B,]
        """
        c = c * context_mask           # [B, 1024] -> [B, 1024]

        # Transformer
        c = self.point_net(c)          # [B, 1024] -> [B, 512]

        t = _t.unsqueeze(1)            
        t = self.time_net(t)          
        t = t.squeeze()               

        g = self.up_net(g)

        g = g.unsqueeze(1)             
        for block in self.blocks:
            g = block(g, c+t)                      # (N, T, D)
        g = g.squeeze(1)
        
        g = self.down_net(g)
        return g