import numpy as np
import torch
import torch.nn as nn

from . import modules as models


from models.components import DiTBlock
class DiTFeatureEncoder(nn.Module):
    def __init__(self, enc_dim=256, latent_size=132, out_dim=7):
        super().__init__()
        latent_size = latent_size
        out_dim = out_dim

        depth       = 4
        enc_dim = enc_dim
        num_heads   = 8
        self.blocks = nn.ModuleList([
            DiTBlock(enc_dim, num_heads) for _ in range(depth)
        ])

        self.time_embed = nn.Sequential(
            models.grasp_dif.GaussianFourierProjection(embed_dim=enc_dim),
            nn.Linear(enc_dim, enc_dim),
            nn.SiLU(),
        )
        self.x_embed = nn.Sequential(
            nn.Linear(3, enc_dim),
            nn.SiLU(),
        )
        self.transform_lin = nn.Linear(latent_size, enc_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(enc_dim, out_dim),
            nn.ReLU(),
        )
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

    def forward(self, x, t, z_embed):
        # x: (B, N, 3)  # N 是点云的点数，在 DiT 里面当做序列长度
        # t: (B, N, 1)     # 时间，其中 N 是 repeat 得到的
        # z_embed: (B, N, D)  # 其中 N 是 repeat 得到的，D 是 latent_size

        if len(x.shape) == 2:  # compute sdf
            x_embed = x.unsqueeze(1)
            x_embed = self.x_embed(x_embed)
            z_embed = self.transform_lin(z_embed)
            
            t_embed = self.time_embed(t)
            z_embed += t_embed

            for block in self.blocks:
                x_embed = block(x_embed, z_embed)                      # (N, T, D)
            
            x_embed = self.out_lin(x_embed)
            x_embed = x_embed.squeeze(1)
            return x_embed
        else:                  # compute noise
            x_embed = self.x_embed(x)
            z_embed = z_embed[:, 0, :]
            z_embed = self.transform_lin(z_embed)
            
            t_embed = self.time_embed(t)
            z_embed += t_embed[:, 0, :]

            for block in self.blocks:
                x_embed = block(x_embed, z_embed)                      # (N, T, D)
            
            x_embed = self.out_lin(x_embed)
            return x_embed


def load_graspdiff(feature_backbone="Default"):
    device = 'cuda'

    class EncoderParams:
        latent_size = 132
        hidden_size = 512
    class FeatureEncoderParams:
        enc_dim = 132
        in_dim = 3
        out_dim = 7
        # dims = [ 512, 512, 512, 512, 512, 512, 512, 512]
        dims = [ 512, 512, 512, 512, 512, 512 ]
        dropout = [0, 1, 2, 3, 4, 5, 6, 7]
        dropout_prob = 0.2
        norm_layers = [0, 1, 2, 3, 4, 5, 6, 7]
        latent_in = [4]
        xyz_in_all = False
        use_tanh = False
        latent_dropout = False
        weight_norm = True
    class DecoderParams:
        hidden_dim = 512
    class PointParams:
        n_points = 30
        loc = [0.0, 0.0, 0.5]
        scale = [0.7, 0.5, 0.7]

    # vision encoder
    vision_encoder = models.vision_encoder.VNNPointnet2(out_features=EncoderParams.latent_size, device=device)
    # Geometry encoder
    geometry_encoder = models.geometry_encoder.map_projected_points
    # Feature Encoder
    if feature_backbone == "DiT":
        feature_encoder = DiTFeatureEncoder(enc_dim=128, latent_size=EncoderParams.latent_size, out_dim=FeatureEncoderParams.out_dim)
    else:
        feature_encoder = models.nets.TimeLatentFeatureEncoder(
            enc_dim=FeatureEncoderParams.enc_dim,
            latent_size=EncoderParams.latent_size,
            dims =FeatureEncoderParams.dims,
            out_dim=FeatureEncoderParams.out_dim,
            dropout=FeatureEncoderParams.dropout,
            dropout_prob=FeatureEncoderParams.dropout_prob,
            norm_layers=FeatureEncoderParams.norm_layers,
            latent_in=FeatureEncoderParams.latent_in,
            xyz_in_all=FeatureEncoderParams.xyz_in_all,
            use_tanh=FeatureEncoderParams.use_tanh,
            latent_dropout=FeatureEncoderParams.latent_dropout,
            weight_norm=FeatureEncoderParams.weight_norm 
        )
    points = models.points.get_3d_pts(n_points = PointParams.n_points,
                        loc=np.array(PointParams.loc),
                        scale=np.array(PointParams.scale))
    in_dim = PointParams.n_points*FeatureEncoderParams.out_dim
    hidden_dim = 512
    energy_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
    )
    model = models.GraspDiffusionFields(vision_encoder=vision_encoder, feature_encoder=feature_encoder, geometry_encoder=geometry_encoder,
                                       decoder=energy_net, points=points).to(device)
    return model