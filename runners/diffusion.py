import os
import logging
import time
import glob
import scipy
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from models.skip import skip
from models.diffusion import Model
from runners.unet import UNet
from utils.data_utils import data_transform, inverse_data_transform
from functions.denoising import efficient_generalized_steps
import torchvision.utils as tvu
from runners.VS2M import VS2M
import random
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, logger, config, image_folder):
        model = VS2M(
            self.args.rank, np.ones((self.config.data.image_size, self.config.data.image_size, self.config.data.channels)),
            np.ones((self.config.data.image_size, self.config.data.image_size, self.config.data.channels)), 
            self.args.beta, self.config.model.iter_number, self.config.model.lr
        )
        self.sample_sequence(model, config, logger, image_folder=image_folder)


    def sample_sequence(self, model, config=None, logger=None, image_folder=None):
        args, config = self.args, self.config
        deg = args.deg

        ## get original msi and mask
        path = os.path.join(config.data.root, config.data.filename)
        mat = scipy.io.loadmat(path)
        
        mask = None
        mat['img_clean'] = mat['img_clean']
        mat['mask_10'] = mat['mask_10']
        mat['mask_20'] = mat['mask_20']
        mat['mask_30'] = mat['mask_30']
        ## get degradation matrix 
        if deg[:10] == 'completion':
            args.sr = int(deg[10:])
            from functions.svd_replacement import Inpainting
            img_clean = torch.from_numpy(np.float32(mat['img_clean'])).permute(2, 0, 1).unsqueeze(0) #（1，32，256，256）
            mask = torch.from_numpy(np.float32(mat['mask_{}'.format(args.sr)])).permute(2, 0, 1).unsqueeze(0) #(256, 256, 32) --->（1，32，256，256）
            mask_vec = mask.clone().reshape(mask.shape[0], mask.shape[1], -1).permute(0, 2, 1).reshape(mask.shape[0], -1)[0,:]  #（1，32，256，256） ---> (1, 32, 65536) ---> (1, 65536, 32) ---> (1, 2097152)
            missing = torch.nonzero(mask_vec == 0).long().reshape(-1) #[1467242]
            keeping = torch.nonzero((1 - mask_vec) == 0).long().reshape(-1) #[629910]
            H_funcs = Inpainting(config.data.channels, config.data.image_size, missing, keeping, self.device)
            mask = np.float32(mat['mask_{}'.format(args.sr)]) #(256, 256, 32)
        
        elif deg[:12] == 'sisr_bicubic':
            factor = int(deg[12:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             config.data.channels, self.config.data.image_size, self.device, stride = factor)
            img_clean = torch.from_numpy(np.float32(mat['img_clean'])).permute(2, 0, 1).unsqueeze(0)

        elif deg[:9] == 'denoising':
            args.sigma_0 = float(deg[9:])
            from functions.svd_replacement import Denoising
            H_funcs = Denoising(config.data.channels, config.data.image_size, self.device)
            img_clean = torch.from_numpy(np.float32(mat['img_clean'])).permute(2, 0, 1).unsqueeze(0)
        
        ## to account for scaling to [-1,1]
        args.sigma_0 = 2 * args.sigma_0 
        sigma_0 = args.sigma_0
        
        x_orig = img_clean
        x_orig = x_orig.to(self.device)

        x_orig = data_transform(self.config, x_orig)

        y_0 = H_funcs.H(x_orig) # (1, 629930) only include the konwn pixel for completion
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)  #add noise on the konwn pixel for completion

        ## in this operation, the known pixels remain unchanged, and the unknown pixels are filled with 0, which is essentially a rearrangement process for completion
        pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size) 
        ## processing the unknown pixel value for completion
        if deg == 'completion':
            pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

        pinv_y_0 = inverse_data_transform(config, pinv_y_0[0,:,:,:]).detach().permute(1,2,0).cpu().numpy()

        x = torch.randn(
            y_0.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        self.sample_image(pinv_y_0, x, model, H_funcs, y_0, sigma_0, mask=mask, img_clean=img_clean, logger=logger, image_folder=image_folder)


    def sample_image(self, pinv_y_0, x, model, H_funcs, y_0, sigma_0, mask=None, img_clean=None, logger=None, image_folder=None):
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        efficient_generalized_steps(pinv_y_0, x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, mask=mask, img_clean = img_clean, logger=logger, args=self.args, config=self.config, image_folder=image_folder)

        
