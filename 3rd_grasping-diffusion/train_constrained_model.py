import os, json
import h5py
import numpy as np
from acronym_tools import load_mesh, load_grasps, create_gripper_marker
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch
import trimesh
import mesh2sdf
from scipy.spatial.transform import Rotation as R
from dataloader.constrained_loader import AcronymAndSDFDataset
from tqdm import tqdm
from positional_embeddings import PositionalEmbedding
import gc
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

# from model import NoiseScheduler, Grasp_Diffusion

class NoiseScheduler():
    def __init__(self,
                 device,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.device = device
        self.num_timesteps = num_timesteps
        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=torch.float32) ** 2
            

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.sqrt_inv_alphas_cumprod = self.sqrt_inv_alphas_cumprod.to(device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)

    def reconstruct_x0(self, x_t, t, noise):
        t = t.to(self.device)
        x_t = x_t.to(self.device)
        noise = noise.to(self.device)
        s1 = self.sqrt_inv_alphas_cumprod[t.to("cpu")]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t.to("cpu")]
        # s1 = s1.reshape(-1, 1)
        # s2 = s2.reshape(-1, 1)
        return s1.to(self.device) * x_t - s2.to(self.device) * noise

    def q_posterior(self, x_0, x_t, t):
        x_0, x_t, t = x_0.to(self.device), x_t.to(self.device), t.to(self.device)
        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        # s1 = s1.reshape(-1, 1)
        # s2 = s2.reshape(-1, 1)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):
        t = t.to("cpu")
        if t[0].sum() == 0:
            return torch.zeros(t.shape)

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)
        return variance

    def step(self, model_output, timestep, sample):
        model_output, timestep, sample = model_output.to(self.device), timestep.to(self.device), sample.to(self.device)
        t = timestep
        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = torch.zeros(t.shape)
        if t[0] > 0:
            noise = torch.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance.to(self.device)

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, timesteps):
        x_start, x_noise, timesteps = x_start.to(self.device), x_noise.to(self.device), timesteps.to(self.device)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)
        # s1 = s1.unsqueeze(0)
        # s2 = s2.unsqueeze(0)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps
    
class Block(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        return x + self.act(self.ff(x))
    
class Block_2(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super().__init__()

        self.ff = nn.Linear(in_size, out_size)
        self.act = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.1)

    def forward(self, x: torch.Tensor):
        return self.act(self.ff(x))

class Conv3DModel(nn.Module):
    """
    Model to create a latent representation of the object
    3 set layers with variable feature map sized
    -- __init__()
    :param in_channels -> number of input channels, for mesh default is 1
    :param level_channels -> array with desired number of feature map sizes for each layer
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, level_channels=[8, 16, 32]) -> None:
        super(Conv3DModel, self).__init__()

        l1, l2, l3, l4 = level_channels[0], level_channels[1], level_channels[2], level_channels[3]

        # Layer 1
        self.conv1_1 = nn.Conv3d(in_channels=2, out_channels=l1, kernel_size=(3,3,3), padding=1)
        
        self.bn1 = nn.BatchNorm3d(num_features=l1)
        # self.conv1_2 = nn.Conv3d(in_channels=l1, out_channels=l1, kernel_size=(3,3,3), padding=1)
        # self.depthwise = nn.Conv3d(in_channels=2, out_channels=l1, kernel_size=(3,3,3), padding=1, groups=2)
        # self.pointwise = nn.Conv3d(in_channels=l1, out_channels=l1, kernel_size=1, padding=0, stride=1)

        self.relu1 = nn.LeakyReLU()
        self.pooling = nn.MaxPool3d(kernel_size=(2,2,2), stride=2)

        # Layer 2
        self.conv2 = nn.Conv3d(in_channels=l1, out_channels=l2, kernel_size=(3,3,3), padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=l2)
        self.relu2 = nn.LeakyReLU()

        # Layer 3
        self.conv3_1 = nn.Conv3d(in_channels=l2, out_channels=l3, kernel_size=(3,3,3), padding=1)
        self.bn3 = nn.BatchNorm3d(num_features=l3)
        self.relu3 = nn.LeakyReLU()
        # self.conv3_2 = nn.Conv3d(in_channels=l3, out_channels=l3, kernel_size=(3,3,3), padding=1)

        # self.bn_bool = False
        self.conv4_1 = nn.Conv3d(in_channels=l3, out_channels=l4, kernel_size=(3,3,3), padding=1)
        self.bn4 = nn.BatchNorm3d(num_features=l4)
        self.relu4 = nn.LeakyReLU()
        # self.conv4_2 = nn.Conv3d(in_channels=l4, out_channels=l4, kernel_size=(3,3,3), padding=1)


    def forward(self, input):
        # res = self.relu(self.bn1(self.conv1_1(input)))
        res = self.relu1(self.conv1_1(input))
        res = self.pooling(res)
        # print("After 1st layer: ", res.shape)

        # res = self.relu(self.bn2(self.conv2(res)))
        res = self.relu2(self.conv2(res))
        res = self.pooling(res)
        # print("After 2nd layer: ", res.shape)
        
        res = self.relu3(self.conv3_1(res))
        # res = self.relu(self.conv3_2(res))
        # res = self.relu(self.bn3(self.conv3_1(res)))
        res = self.pooling(res)
        # print("After 3rd layer: ", out.shape)

        out = self.relu4(self.conv4_1(res))
        # out = self.relu(self.bn4(self.conv4_1(res)))
        # res = self.relu(self.conv4_2(res))
        # out = self.pooling(out)
        
        return out
    
class Linear_Block(nn.Module):
    def __init__(self, input_dim) -> None:
        super(Linear_Block, self).__init__()

        layers = []
        layers.append(Block_2(input_dim, 2048))
        layers.append(Block(2048))
        layers.append(Block(2048))
        layers.append(nn.Linear(2048, 7))

        self.linear = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.linear(input)


class Grasp_Diffusion(nn.Module):
    """
    The Grasp Diffusion model conditioned on a 3d voxel grid
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input ->
        1. object_input -> voxel grid representation of size (32, 32, 32)
        2. grasp_input -> tensor containing information of septernion (position + quaternion) of size (1, 7)
        3. timsteps -> tensor containing random timestep of size (1,)
    :return -> Tensor
    """
    
    def __init__(self, device, time_emb=64, level_channels=[8, 16, 32, 64]) -> None:
        super(Grasp_Diffusion, self).__init__()
        self.device = device

        self.conv_model = Conv3DModel(level_channels=level_channels)
        self.conv_model = self.conv_model.to(device)

        self.time_mlp = PositionalEmbedding(time_emb, "sinusoidal")
        grasp_emb_size = 64
        self.input_mlp = PositionalEmbedding(grasp_emb_size, "sinusoidal", scale=25.0)

        self.scale_dim = 64
        self.scale_embedding = PositionalEmbedding(self.scale_dim, "sinusoidal")
        self.scale_embedder = Block(self.scale_dim)

        latent_mesh_repr_size = level_channels[-1] * 4**3

        grasp_repr_size = 7 * grasp_emb_size
        # concat_size = len(self.time_mlp.layer) + hidden_dim_mesh + grasp_repr_size
        concat_size = time_emb + latent_mesh_repr_size + grasp_repr_size + self.scale_dim

        # self.multihead_attn = nn.MultiheadAttention(embed_dim=latent_mesh_repr_size, num_heads=2)
        
        self.joint_mlp = Linear_Block(concat_size).to(device)

    def forward(self, object_input, grasp_input, timesteps, scale):
        device = self.device

        out = self.conv_model(object_input)

        mesh_repr_flattened = nn.Flatten(1)(out)

        time_ = self.time_mlp(timesteps.to("cpu"))
        time_ = time_.float().to(device)

        x_emb = self.input_mlp(grasp_input.to("cpu"))
        x_emb = nn.Flatten(1)(x_emb)
        x_emb = x_emb.float().to(device)

        scale_emb = self.scale_embedding(scale.to("cpu")).float()
        scale_emb = self.scale_embedder(scale_emb.to(device))
        scale_emb = scale_emb.float().to(device)
        x = torch.cat((x_emb, time_, mesh_repr_flattened.float().to(device), scale_emb), dim=-1)

        x = self.joint_mlp(x)

        return x
    
# Enter directory path of CONG dataset
data_dir = "/home/username/data/v3_constrained/"

grasp_dict_file = os.path.join(data_dir, 'constrained_grasps_split.json')
grasp_data = json.load(open(grasp_dict_file, "r"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device used for training: ",device)
print(" ----------- \n")

print("Loading dataset ...")
train_dataset = AcronymAndSDFDataset(grasp_data, loader_type="train", data_dir=data_dir)

batch_size_num = 32

# , pin_memory=True, pin_memory_device="cuda"
train_dataloader = DataLoader(train_dataset, batch_size=batch_size_num, shuffle=True, num_workers=4)

print("Length of Dataset: ", len(train_dataset), " Number of Batches: ", len(train_dataloader))
print(" ----------- \n")

noise_scheduler = NoiseScheduler(device, num_timesteps=1000)
model = Grasp_Diffusion(device, time_emb=64, level_channels=[16, 32, 64, 64])
model.to(device)
print()

from torchinfo import summary

input_size1 = torch.Size([32, 2, 32, 32, 32])
input_size2 = torch.Size([32, 7])
input_size3 = torch.Size([32])
input_size4 = torch.Size([32])

# Assuming model is your model instance and it takes three inputs
print(summary(model, input_size=[(input_size1), (input_size2), (input_size3), (input_size4)]))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

PATH = "/home/username/constrained_training/training/checkpoints/v1/constrained_model.pt"
model.load_state_dict(torch.load(PATH)['model_state_dict'])

optimizer.load_state_dict(torch.load(PATH)['optimizer_state_dict'])
# scheduler.load_state_dict(torch.load(PATH)['scheduler_state_dict'])

# global_step = 0
frames = []
losses = []
step_loss = []

global_step = torch.load(PATH)['step']

mid_time_losses = []

for param_group in optimizer.param_groups:
    print("LEARNING RATE: ", param_group['lr'])

scheduler = ReduceLROnPlateau(optimizer, patience=1200, verbose=True, factor=0.85, min_lr=1e-7)
scheduler.load_state_dict(torch.load(PATH)['optimizer_state_dict'])

curr_epoch_num = 0
print("Starting Epoch number: ", curr_epoch_num)

with open("/home/username/constrained_training/training/checkpoints/v1/loss_step.json") as f:
    step_loss = json.load(f)

print("Training model ...")
print(" ----------- \n")
model.train()
for epoch in range(10000):
    print("EPOCH NUMBER: ", epoch)
    print(" ----------- \n")
    print(" ----------- \n")
    temp_losses = []
    # pbar = tqdm(train_dataloader)
    for step, (Septernion, sdf, sdf_scale) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()
        noise = torch.randn(Septernion.shape)
        timesteps = torch.randint(
                    0, noise_scheduler.num_timesteps, (Septernion.shape[0],)
                ).long()
        sdf = sdf.to(device)
        Septernion = Septernion.to(device)
        noise = noise.to(device)
        timesteps = timesteps.to(device)
        noisy = noise_scheduler.add_noise(Septernion, noise, timesteps)
        sdf_scale = sdf_scale.to(device)

        noisy = noisy.to(device)

        noise_pred = model(sdf, noisy, timesteps, sdf_scale)
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)

        optimizer.step()
        losses.append(loss.detach().item())

        mid_time_losses.append(loss.detach().item())
        if ((global_step % (1000//batch_size_num)) == 0 and global_step > 0):
            val = np.mean(np.array(mid_time_losses))
            print("Step: ", global_step, "  Train Loss: ", val, 'learning rate:', optimizer.param_groups[0]['lr'])
            scheduler.step(val)
            print("\n")
            mid_time_losses = []
            step_loss.append(val)
            # break;

        if (global_step % (100000//batch_size_num) == 0 and global_step > 0):
            PATH = os.path.join("/home/username/constrained_training/training/checkpoints/v1", "constrained_model.pt")
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'loss': loss.detach().item(),
                'epoch':epoch
                }, PATH)
            Loss_step_PATH = os.path.join("/home/username/constrained_training/training/checkpoints/v1", "loss_step.json")
            with open(Loss_step_PATH, "w") as f:
                json.dump(step_loss, f)
            print("saved")

        global_step += 1

        # break;
        gc.collect()
        torch.cuda.empty_cache()