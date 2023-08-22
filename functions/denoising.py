import torch
from tqdm import tqdm
import torchvision.utils as tvu
import os
import numpy as np
import scipy.io as scio
from runners.com_psnr import quality
from utils.data_utils import inverse_data_transform

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def efficient_generalized_steps(pinv_y_0, x, seq, model, b, H_funcs, y_0, sigma_0, etaB, etaA, etaC, mask=None, img_clean = None, logger=None, args=None, config=None, image_folder=None):
    ## prepare some vectors used in the algorithm
    singulars = H_funcs.singulars() 
    Sigma = torch.zeros(x.shape[1]*x.shape[2]*x.shape[3], device=x.device)
    Sigma[:singulars.shape[0]] = singulars
    U_t_y = H_funcs.Ut(y_0)
    Sig_inv_U_t_y = U_t_y / singulars[:U_t_y.shape[-1]]

    largest_alphas = compute_alpha(b, (torch.ones(x.size(0)) * seq[-1]).to(x.device).long())
    largest_sigmas = (1 - largest_alphas).sqrt() / largest_alphas.sqrt()
    large_singulars_index = torch.where(singulars * largest_sigmas[0, 0, 0, 0] > sigma_0)
    inv_singulars_and_zero = torch.zeros(x.shape[1] * x.shape[2] * x.shape[3]).to(singulars.device)
    inv_singulars_and_zero[large_singulars_index] = sigma_0 / singulars[large_singulars_index]
    inv_singulars_and_zero = inv_singulars_and_zero.view(1, -1)

    ## initialize some variables in the diffusion process
    init_y = torch.zeros(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]).to(x.device)
    init_y[:, large_singulars_index[0]] = U_t_y[:, large_singulars_index[0]] / singulars[large_singulars_index].view(1, -1)
    init_y = init_y.view(*x.size())
    remaining_s = largest_sigmas.view(-1, 1) ** 2 - inv_singulars_and_zero ** 2
    remaining_s = remaining_s.view(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).clamp_min(0.0).sqrt()
    init_y = init_y + remaining_s * x
    init_y = init_y / largest_sigmas

    ## setup iteration variables
    x = H_funcs.V(init_y.view(x.size(0), -1)).view(*x.size())
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])

    iii= 0
    psnr_best = 0
    best = None
    ## iterate over the timesteps
    for i, j in tqdm(zip(reversed(seq), reversed(seq_next))):
        iii +=1
        with torch.no_grad():
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            if iii == 1:
                xt = x.to(x.device)
                avg = np.array(0)
            else:
                xt = xt_next.to(x.device)
                avg = x0_t[0, :,:, :].permute(1,2,0).cpu().numpy()

        ## The untrained network parameters are not updated until iii>args.start_point, roughly equivalent to starting the backdiffusion process from the args.start_point step
        update = False if iii < args.start_point else True
        if update: 
            x0_t, step, best_, psnr_best_= model.optimize(xt.squeeze().permute(1, 2, 0).detach().cpu().numpy(),
                            img_clean.squeeze().permute(1, 2, 0).detach().cpu().numpy(), at, mask, config.model.iter_number[iii-1], logger, avg, update) 
            x0_t = torch.from_numpy(x0_t).permute(2, 0, 1).unsqueeze(0).cuda()
            x0_t = x0_t * 2 - 1.0
            if psnr_best_ > psnr_best:
                psnr_best = psnr_best_
                best = best_
        else:
            x0_t = xt

        ## estimate the diffusion noise contained in xt
        et = (xt - at.sqrt() * x0_t) / (1 - at).sqrt()

        ## variational inference conditioned on y (Eqn. 13 in the main paper)
        with torch.no_grad():
            sigma = (1 - at).sqrt()[0, 0, 0, 0] / at.sqrt()[0, 0, 0, 0]
            sigma_next = (1 - at_next).sqrt()[0, 0, 0, 0] / at_next.sqrt()[0, 0, 0, 0]
            V_t_x0 = H_funcs.Vt(x0_t)
            SVt_x0 = (V_t_x0 * Sigma)[:, :U_t_y.shape[1]]

            falses = torch.zeros(V_t_x0.shape[1] - singulars.shape[0], dtype=torch.bool, device=xt.device)
            cond_before_lite = singulars * sigma_next > sigma_0
            cond_after_lite = singulars * sigma_next < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            std_nextC = sigma_next * etaC
            sigma_tilde_nextC = torch.sqrt(sigma_next ** 2 - std_nextC ** 2)

            std_nextA = sigma_next * etaA
            sigma_tilde_nextA = torch.sqrt(sigma_next**2 - std_nextA**2)

            diff_sigma_t_nextB = torch.sqrt(sigma_next ** 2 - sigma_0 ** 2 / singulars[cond_before_lite] ** 2 * (etaB ** 2))

            Vt_xt_mod_next = V_t_x0 + sigma_tilde_nextC * H_funcs.Vt(et) + std_nextC * torch.randn_like(V_t_x0)

            #less noisy than y (after)
            Vt_xt_mod_next[:, cond_after] = \
                V_t_x0[:, cond_after] + sigma_tilde_nextA * ((U_t_y - SVt_x0) / sigma_0)[:, cond_after_lite] + std_nextA * torch.randn_like(V_t_x0[:, cond_after])

            #noisier than y (before)
            Vt_xt_mod_next[:, cond_before] = \
                (Sig_inv_U_t_y[:, cond_before_lite] * etaB + (1 - etaB) * V_t_x0[:, cond_before] + diff_sigma_t_nextB * torch.randn_like(U_t_y)[:, cond_before_lite])

            #aggregate all cases and give next prediction
            xt_mod_next = H_funcs.V(Vt_xt_mod_next)
            xt_next = (at_next.sqrt()[0, 0, 0, 0] * xt_mod_next).view(*x.shape)

            x0_t = torch.clamp((x0_t + 1.0) / 2.0, 0.0, 1.0)
            psnr = quality(x0_t.squeeze().cpu().permute(1,2,0).numpy(), img_clean.squeeze().permute(1,2,0).numpy())

            if psnr_best < psnr:
                psnr_best = psnr
                best = x0_t.squeeze().cpu().permute(1,2,0).numpy()
            logger.info('iteration: {}, psnr: {}, psnr_best: {}'.format(iii, psnr, psnr_best))

    x_ = inverse_data_transform(config, xt_next.to('cpu'))

    x_save = x_[0,:,:,:].permute(1,2,0).cpu().numpy()
    x_best = best
    img_clean_save = img_clean[0,:,:,:].permute(1,2,0).cpu().numpy()
    psnr = quality(x_save, img_clean[0,:,:,:].permute(1,2,0).detach().cpu().numpy())

    scio.savemat(
        os.path.join(image_folder, f"x_{iii}.mat"), {'y_0': pinv_y_0, 'x_recon':x_save, 'img_clean':img_clean_save, 'psnr':psnr, 'x_best':np.clip(x_best,0,1), 'psnr_best':psnr_best}
    )



