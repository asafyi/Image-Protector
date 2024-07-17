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

def init_models(gan_enc_loc,dif_model,enc_only=False):
    """
    initiate the models to be attacked
    :param gan_enc_loc: the path to the styleGAN model
    :param dif_model: the diffusion model to be used
    :param enc_only: if true will return only the encoder of the styleGAN model and no the entire model
    :return: None
    """
    print("---LOADING MODELS (this may take a little time)---")
    GAN_net, _ = setup_model(gan_enc_loc,'cuda',encoder_only=enc_only)
    if dif_model == 'I2I':
        DIF_model = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",revision="fp16", torch_dtype=torch.float16).to('cuda')
    if dif_model == 'IN':
        DIF_model = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting",revision="fp16",torch_dtype=torch.float16).to('cuda')
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


def save_image(img, save_dir, name):
    """
    Save an image to the directory = save_dir with name = name
    :param img: PIL Image
    :param save_dir: the directory the image is to be saved to
    :param name: the name of the image
    :return: None
    """
    im_save_path = os.path.join(save_dir, name)
    img.save(im_save_path)


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


def save_args(args,save_dir):
    """
    saves the values of args to a file in save_dir
    :param args: the args
    :param save_dir: the save dir
    :return: None
    """
    with open(os.path.join(save_dir,'attack arguments.txt'),'w') as wrt:
        for arg in vars(args):
            try:
                argument = 'argument: ' + arg + ' is: ' + str(getattr(args,arg)) + '\n'
            except:
                argument = 'argument: ' + arg + ' is: None\n'
            wrt.write(argument)
        wrt.close()


def run_attack(args,DIF_model,GAN_net):
    print("---CREATING DIRS---")
    #open all the directories for the results of the attack
    image_dir = Path(args.im_dir)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)

    save_args(args,save_dir)

    save_dir.joinpath('original').mkdir(exist_ok=True, parents=True)
    save_dir.joinpath('with_noise').mkdir(exist_ok=True, parents=True)

    inversions_directory_path = os.path.join(args.save_dir, 'benign_GAN_inversions')
    os.makedirs(inversions_directory_path, exist_ok=True)

    adversarial_inversions_directory_path = os.path.join(args.save_dir, 'adversarial_GAN_inversions')
    os.makedirs(adversarial_inversions_directory_path, exist_ok=True)

    if not args.prompt == None:
        save_dir.joinpath('gen_image_original').mkdir(exist_ok=True, parents=True)
        save_dir.joinpath('gen_image_adversarial').mkdir(exist_ok=True, parents=True)

    print("---FINISHED CREATING DIRS---")

    #start iterating through the images in the image_dir directory
    restyle = (args.gan_enc == 'restyle')
    for img_path in image_dir.iterdir():

        img = Image.open(img_path)
        img_name = img_path.name
        print("---NOW RUNNING ON: " + img_name + "---")

        # preprocess the image and save the croped version
        original_path = Path(save_dir).joinpath('original')
        X = preprocess(img, 512)
        save_image(tensor2im(X[0]),original_path,img_name)
        X = X.to('cuda')

        #run the actual attack
        if args.attack_type == 'comb':
            X_adv = pgd(X,DIF_model.vae.encode,GAN_net,restyle, args.epsilon,args.pgd_iters,args.alpha,args.step_size,args.loss_type)
            X_adv = tensor2im(X_adv[0])

        elif args.attack_type == 'sep':
            #attack DIF
            X_dif_adv = pgd2(X, DIF_model.vae.encode, 'DIF', restyle, args.epsilon, args.pgd_iters, args.step_size)

            #attack GAN
            #extract faces
            faces_lst, loc_lst = align_face(tensor2im(X[0]))
            faces_lst_adv = []
            for face in faces_lst: #attack each face seperatly
                X_face = preprocess(face,256)
                X_face = X_face.to('cuda')
                X_face_adv = pgd2(X_face,GAN_net,'GAN',restyle, args.epsilon, args.pgd_iters, args.step_size)
                faces_lst_adv.append(X_face_adv)
                del X_face #free GPU space

            #transform the tensors to images
            X_dif_adv = tensor2im(X_dif_adv[0])
            faces_lst_adv_fixed = [tensor2im(face[0]) for face in faces_lst_adv]
            #paste faces into bigger image
            X_adv = embed_faces(X_dif_adv, faces_lst_adv_fixed, loc_lst)

        else:
            print('not a valid attack_type : ', args.attack_type)
            return


        save_image(X_adv,save_dir.joinpath('with_noise'),img_name)   #save the resulting image of the attack

        X_adv = preprocess(X_adv, 512)
        X_adv.to('cuda')
        generator = GAN_net.decoder #init the styleGAN generator
        generator.eval()

        print('---CALCULATING STYLEGAN OUTPUT---')
        #run the styleGAN encoder on the original image and save the result
        X_gan = swap_model(X,'gan')
        original_latent = get_GAN_latents(GAN_net, X_gan, restyle)
        original_imgs_inv, _ = generator([original_latent], input_is_latent=True, randomize_noise=False,
                                return_latents=True)
        save_image(tensor2im(original_imgs_inv[0]), inversions_directory_path, img_name)

        #run the styleGAN encoder on the perturbed image and save the result
        X_adv_gan = swap_model(X_adv,'gan')
        X_adv_gan = X_adv_gan.to('cuda')
        adv_latent = get_GAN_latents(GAN_net, X_adv_gan, restyle)
        imgs_inv, _ = generator([adv_latent], input_is_latent=True, randomize_noise=False,
                                return_latents=True)
        save_image(tensor2im(imgs_inv[0]), adversarial_inversions_directory_path, img_name)


        #if the user entered a prompt we run the diffusion models also
        if not args.prompt == None:
            print('---CALCULATING DIFFUSION MODEL OUTPUT---')
            prompt = args.prompt
            SEED = np.random.randint(low=0, high=10000)
            if not args.seed == None:
                SEED = args.seed

            #these parameters can be tweaked
            STRENGTH = 0.5
            GUIDANCE = 7.5
            NUM_STEPS = 50

            with torch.autocast('cuda'):
                #run the diffusion model on the original image and the prompt and save the result
                torch.manual_seed(SEED)
                X = fix_for_diffusion(X)
                init_image = to_pil(X).convert("RGB")
                image_nat = DIF_model(prompt=prompt, image=init_image, strength=STRENGTH, guidance_scale=GUIDANCE,
                                         num_inference_steps=NUM_STEPS).images[0]
                image_nat.save(save_dir.joinpath('gen_image_original/' + img_name), 'PNG')

                # run the diffusion model on the perturbed image and the prompt and save the result
                torch.manual_seed(SEED)
                X_adv = fix_for_diffusion(X_adv)
                adv_image = to_pil(X_adv).convert("RGB")
                image_adv = DIF_model(prompt=prompt, image=adv_image, strength=STRENGTH, guidance_scale=GUIDANCE,
                                         num_inference_steps=NUM_STEPS).images[0]
                image_adv.save(save_dir.joinpath('gen_image_adversarial/' + img_name), 'PNG')

        #free GPU space
        del X
        del X_adv
        torch.cuda.empty_cache()


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








if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main attack script")

    parser.add_argument("--im_dir", type=str, default='./images', help='directory containing the original images')
    parser.add_argument("--save_dir", type=str, default='./attack_output', help='directory for the result of the attack')
    parser.add_argument("--gan_enc", type=str, default='e4e', help='e4e/psp/restyle')
    parser.add_argument("--dif_model", type=str, default='I2I', help='I2I/IN')
    parser.add_argument("--attack_type", type=str, default='comb', help='attack type to be used. sep/comb')

    parser.add_argument("--prompt", type=str, default=None, help='enter an optional prompt for the diffuser')
    parser.add_argument("--seed", type=int, default=None, help='seed for diffusion image gen')


    parser.add_argument('--pgd_iters', type=int, default=100, help='number of pgd iterations to run')
    parser.add_argument('--epsilon', type=float, default=0.04, help='maximum size of noise')
    parser.add_argument('--loss_type', type=str, default='adp', help='adp/convex')
    parser.add_argument('--alpha', type=float, default=0.1, help='weight for diffusion/GAN loss calculations')
    parser.add_argument('--step_size', type=float, default=0.05, help='step size in pgd')

    args = parser.parse_args()
    assert args.gan_enc in {'e4e', 'psp', 'restyle'} and args.dif_model in {'I2I', 'IN'}
    assert os.path.isdir(args.im_dir)
    assert args.pgd_iters >= 0 and args.epsilon>0
    assert args.alpha>=0 and args.alpha<=1
    assert args.step_size>0


    gan_enc = paths_config.model_paths[args.gan_enc]
    DIF_model,GAN_net = init_models(gan_enc,args.dif_model)
    run_attack(args,DIF_model,GAN_net)




