import argparse

import torch
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(".")
sys.path.append("..")
sys.path.append("../stylegan_model")

import matplotlib.pyplot as plt
from configs import data_configs, paths_config
#from datasets.inference_dataset import InferenceDataset
#from torch.utils.data import DataLoader
from utils.model_utils import setup_model
from utils.common import tensor2im
from utils.alignment import align_face, embed_faces
from torch.nn import functional as F
#from criteria.id_loss import IDLoss2

from PIL import Image, ImageOps

from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
#from utils import preprocess, recover_image

to_pil = T.ToPILImage()


def init_models():
    """
    initiate the models to be attacked StyleGAN e4e and stable diffusion v1.5
    :return: the diffusion model and stylegan model
    """
    print("---LOADING MODELS (this may take a little time)---")
    gan_enc_loc = paths_config.model_paths["e4e"]
    GAN_net, _ = setup_model(gan_enc_loc,'cuda',encoder_only=False)
    DIF_model = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",revision="fp16", torch_dtype=torch.float16).to('cuda')
    print("---FINISHED LOADING MODELS---")
    return DIF_model, GAN_net


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
    if image.ndim != 3:
        print("please use a non black-white image")
        exit(1)
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.0 * image - 1.0
    return image



def get_GAN_latents(GAN_net, X, restyle = False):
    """
    Find the encoding of the image X by the styleGAN encoder of GAN_net, i.e. return Enc(X) where
    Enc is the encoder of GAN_net
    :param GAN_net: the styleGAN model
    :param X: the image, should be processed os size (1,3,A,B)
    :param restyle: necessary for technical reasons
    :return: the encoded image
    """
    if restyle:
        X = torch.cat([X, GAN_net.avg_image], dim=1)
    codes = GAN_net.encoder(X)
    if GAN_net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + GAN_net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + GAN_net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes


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




def run_attack(img, DIF_model, GAN_net):
    #start iterating through the images in the image_dir directory
    img = img.convert('RGB')
    restyle = False
    attack_type = 'comb'
    epsilon = 0.04
    pgd_iters = 100
    alpha = 0.1
    step_size = 0.05
    loss_type = 'adp'
    print("---NOW RUNNING  attack ---")

    # preprocess the image and run the actual attack
    X = preprocess(img, 512)
    X = X.to('cuda')
    if attack_type == 'comb':
        X_adv = pgd(X,DIF_model.vae.encode, GAN_net, restyle, epsilon, pgd_iters, alpha, step_size, loss_type)
        X_adv = tensor2im(X_adv[0])

    elif attack_type == 'sep':
        #attack DIF
        X_dif_adv = pgd2(X, DIF_model.vae.encode, 'DIF', restyle, epsilon, pgd_iters, step_size)

        #attack GAN
        #extract faces
        faces_lst, loc_lst = align_face(tensor2im(X[0]))
        faces_lst_adv = []
        for face in faces_lst: #attack each face seperatly
            X_face = preprocess(face,256)
            X_face = X_face.to('cuda')
            X_face_adv = pgd2(X_face,GAN_net,'GAN',restyle, epsilon, pgd_iters, step_size)
            faces_lst_adv.append(X_face_adv)
            del X_face #free GPU space

        #transform the tensors to images
        X_dif_adv = tensor2im(X_dif_adv[0])
        faces_lst_adv_fixed = [tensor2im(face[0]) for face in faces_lst_adv]
        #paste faces into bigger image
        X_adv = embed_faces(X_dif_adv, faces_lst_adv_fixed, loc_lst)

    #free GPU space
    del X
    torch.cuda.empty_cache()
    return X_adv



def pgd(X, DIF_encoder, GAN_net, restyle, eps, iters, alpha, step_size,loss_type):
    #init noise and X for the gan and dif models
    noise = 2*eps*torch.rand_like(X) - eps
    X_gan = swap_model(X,'gan')
    X_dif = swap_model(X,'dif')
    X_GAN_enc = get_GAN_latents(GAN_net,X_gan,restyle).detach()
    X_DIF_enc = DIF_encoder(X_dif).latent_dist.mean.detach()

    #start pgd
    pbar = tqdm(range(iters))
    X_adv = None
    for i in pbar:
        actual_step_size = step_size/(np.sqrt(i+1))
        X_adv = torch.clamp(X+noise,min=-1,max=1) #update X_adv
        X_adv.requires_grad = True

        #make X_adv ok for gan and dif models
        Y_adv_gan = swap_model(X_adv,'gan')
        Y_adv_dif = swap_model(X_adv,'dif')

        #calculate the encoding by the gan and dif encoders
        X_adv_GAN_enc = get_GAN_latents(GAN_net,Y_adv_gan,restyle)
        X_adv_DIF_enc = DIF_encoder(Y_adv_dif).latent_dist.mean

        delta_gan = F.l1_loss(X_GAN_enc, X_adv_GAN_enc)
        delta_dif = F.l1_loss(X_DIF_enc, X_adv_DIF_enc)

        if loss_type == 'convex': #this loss is a convex combination alpha*x+(1-alpha)*y
            loss = -alpha * delta_gan - (1-alpha) * delta_dif
        if loss_type == 'adp':  #this loss function is xln(1+y)+yln(1+x)
            loss = -alpha*delta_gan * torch.log(1+delta_dif) - (1-alpha)*delta_dif * torch.log(1+delta_gan)

        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        loss.backward() #backpropagate

        #update noise
        noise = noise - actual_step_size * X_adv.grad.sign()
        noise = torch.clamp(noise, -eps, eps)

    return X_adv



def pgd2(X, Enc, Enc_type, restyle, eps, iters, step_size):
    #make X ok for the model
    if Enc_type == 'DIF':
        X = swap_model(X,'dif')
        X_enc = Enc(X).latent_dist.mean.detach()
    if Enc_type == 'GAN':
        X = swap_model(X,'gan')
        X_enc = get_GAN_latents(Enc,X,restyle).detach()

    #init noise and X_adv
    noise = 2 * eps * torch.rand_like(X) - eps
    pbar = tqdm(range(iters))
    X_adv = None
    for i in pbar:
        actual_step_size = step_size / (np.sqrt(i + 1))
        X_adv = torch.clamp(X + noise, min=-1, max=1)  # update X_adv
        X_adv.requires_grad = True

        #get encoding of X_adv
        if Enc_type == 'DIF':
            X_adv_enc = Enc(X_adv).latent_dist.mean
        if Enc_type == 'GAN':
            X_adv_enc = get_GAN_latents(Enc,X_adv,restyle)

        #calc loss
        loss = -F.l1_loss(X_enc,X_adv_enc)
        pbar.set_description(f"[Running attack]: Loss {loss.item():.5f} | step size: {actual_step_size:.4}")
        loss.backward() #backpropagate

        # update noise
        noise = noise - actual_step_size * X_adv.grad.sign()
        noise = torch.clamp(noise, -eps, eps)

    return X_adv
