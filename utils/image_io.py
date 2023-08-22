import glob

import torch
import torchvision
import numpy as np
from PIL import Image
# import skvideo.io



def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()


"""
    hs_io
"""

def get_noisy_image(img_np, sigma, percetage):
    """Adds Gaussian noise to an image.

    Args:
        img_np: image, np.array with values from 0 to 1
        sigma: std of the noise
    """
    img_noisy_np = np.clip(img_np + np.random.normal(scale=sigma, size=img_np.shape), 0, 1).astype(np.float32)

    NoiseNum = int(percetage * img_noisy_np.shape[0] * img_noisy_np.shape[1] * img_noisy_np.shape[2])
    for i in range(NoiseNum):
        randX = np.random.random_integers(0, img_noisy_np.shape[0] - 1)
        randY = np.random.random_integers(0, img_noisy_np.shape[1] - 1)
        randZ = np.random.random_integers(0, img_noisy_np.shape[2] - 1)
        if np.random.random_integers(0, 1) <= 0.5:
            img_noisy_np[randX, randY, randZ] = 0
        else:
            img_noisy_np[randX, randY, randZ] = 1
    return img_noisy_np


def load_image(file_name):
    mat = scipy.io.loadmat(file_name)
    img_np = mat["image"]
    img_np = img_np.transpose(2, 0, 1)
    return img_np




