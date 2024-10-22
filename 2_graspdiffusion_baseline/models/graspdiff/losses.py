import torch
import torch.nn as nn


class SDFLoss():
    def __init__(self, field='sdf', delta = 0.6, grad=True):
        self.field = field
        self.delta = delta

        self.grad = grad

    def loss_fn(self, model, model_input, ground_truth, val=False):
        loss_dict = dict()
        label = ground_truth[self.field].squeeze().reshape(-1)

        ## Set input ##
        x_sdf = model_input['x_sdf'].detach().requires_grad_()
        c = model_input['visual_context']

        ## Compute model output ##
        model.set_latent(c, batch=x_sdf.shape[1])
        sdf = model.compute_sdf(x_sdf.view(-1, 3))

        ## Reconstruction Loss ##
        loss = nn.L1Loss(reduction='mean')
        pred_clip_sdf = torch.clip(sdf, -10., self.delta)
        target_clip_sdf = torch.clip(label, -10., self.delta)
        l_rec = loss(pred_clip_sdf, target_clip_sdf)

        ## Total Loss
        loss_dict[self.field] = l_rec

        info = {'sdf': sdf}
        return loss_dict, info


class ProjectedSE3DenoisingLoss():
    def __init__(self, field='denoise', delta = 1., grad=False):
        self.field = field
        self.delta = delta
        self.grad = grad

    # TODO check sigma value
    def marginal_prob_std(self, t, sigma=0.5):
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    # @note diffusion 损失函数
    def loss_fn(self, model, model_input, ground_truth, val=False, eps=1e-5):

        ## Set input ##
        H = model_input['x_ene_pos']
        H_prior = model_input['x_ene_pos_prior']
        c = model_input['visual_context']
        model.set_latent(c, batch=H.shape[1])
        H = H.reshape(-1, 4, 4)
        H_prior = H_prior.reshape(-1, 4, 4)
        model.set_prior_H_latent(H_prior)

        ## 1. H to vector ##
        H_th = SO3_R3(R=H[...,:3, :3], t=H[...,:3, -1])
        xw = H_th.log_map()

        ## 2. Sample perturbed datapoint ##
        random_t = torch.rand_like(xw[...,0], device=xw.device) * (1. - eps) + eps
        z = torch.randn_like(xw)
        std = self.marginal_prob_std(random_t)
        perturbed_x = xw + z * std[..., None]
        perturbed_x = perturbed_x.detach()
        perturbed_x.requires_grad_(True)

        ## Get gradient ##
        with torch.set_grad_enabled(True):
            perturbed_H = SO3_R3().exp_map(perturbed_x).to_matrix()
            energy = model(perturbed_H, random_t)
            grad_energy = torch.autograd.grad(energy.sum(), perturbed_x,
                                              only_inputs=True, retain_graph=True, create_graph=True)[0]

        # Compute L1 loss
        z_target = z/std[...,None]
        loss_fn = nn.L1Loss()
        loss = loss_fn(grad_energy, z_target)/10.

        info = {self.field: grad_energy}
        loss_dict = {"Score loss": loss}
        return loss_dict, info


class LossDictionary():

    def __init__(self, loss_dict):
        self.fields = loss_dict.keys()
        self.loss_dict = loss_dict

    def loss_fn(self, model, model_input, ground_truth, val=False):

        losses = {}
        infos = {}
        for field in self.fields:
            loss_fn_k = self.loss_dict[field]
            loss, info = loss_fn_k(model, model_input, ground_truth, val)
            losses = {**losses, **loss}
            infos = {**infos, **info}

        return losses, infos


def get_loss_fn():
    loss_dict = {
       'denoise': ProjectedSE3DenoisingLoss(),
       'sdf': SDFLoss(),
    }
    return LossDictionary(loss_dict)
