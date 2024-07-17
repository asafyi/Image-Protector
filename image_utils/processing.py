import torch
import numpy as np
import os
import sys

sys.path.append(".")
sys.path.append("..")
import torchvision.transforms as T
from stylegan_model.utils.common import tensor2im
from PIL import Image



def preprocess(image,size=256):
    """
    preprocess an image for the attack
    :param image: an image
    :param size: image will be resized to a sizeXsize image
    :return: the preprocessed image
    """
    resize = T.transforms.Resize(size)
    center_crop = T.transforms.CenterCrop(size)
    image = center_crop(resize(image))
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.0 * image - 1.0
    return image


def save_image(img, save_dir, name):
    """
    Save an image to the directory = save_dir with name = name
    :param img: the image to be saved, should be processed, i.e. have entries in (-1,1) and a 3-dim tensor of shape (3,A,B)
    :param save_dir: the directory the image is to be saved to
    :param name: the name of the image
    :return: None
    """
    result = tensor2im(img)
    im_save_path = os.path.join(save_dir, name)
    Image.fromarray(np.array(result)).save(im_save_path)


def fix_for_diffusion(X):
    """
    Takes a processed image X of size (1,3,A,B) with entries in (-1,1)
    and reverts it back to a (3,A,B) image with entries in (0,1)
    :param X: the image
    :return: the reverted image
    """
    resize = T.transforms.Resize(512)
    X = resize(X[0])
    return (X/2 + 0.5)



def swap_model(X,model):
    """
    Takes a processed image X and manipulates it so that it is a valid input
    for the model - model
    :param X: the image
    :param model: the model, should be either 'dif' or 'gan'
    :return: the manipulated image
    """
    if model == 'dif':
        resize = T.transforms.Resize(512)
        Y = resize(X)
        Y = Y.to(torch.float16)
    if model == 'gan':
        resize = T.transforms.Resize(256)
        Y = resize(X)
        Y = Y.to(torch.float32)
    return Y