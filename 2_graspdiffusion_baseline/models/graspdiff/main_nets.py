import numpy as np
import torch
import torch.nn as nn

from . import modules as models


def load_graspdiff():
    device = 'cuda'

    class EncoderParams:
        latent_size = 132
        hidden_size = 512
    class FeatureEncoderParams:
        enc_dim = 132
        in_dim = 3
        out_dim = 7
        dims = [ 512, 512, 512, 512, 512, 512, 512, 512]
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