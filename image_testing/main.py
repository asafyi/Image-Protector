import argparse
import torch
import numpy as np
import sys
import os
import clip
from pathlib import Path
from PIL import Image
import diffusers
import torchvision.transforms as T

sys.path.append(".")
sys.path.append("..")
sys.path.append("../stylegan_model")

from configs import paths_config
from utils.model_utils import setup_model
from utils.common import tensor2im

from image_utils.alignment import *
from image_utils.processing import *
from styleclip.manipulate import Manipulator
from styleclip.StyleCLIP import GetDt,GetBoundary



to_pil = T.ToPILImage()
model_clip, _ = clip.load("ViT-B/32", device="cuda",jit=False)

network_pkl='../stylegan_model/styleclip/model/stylegan2-ffhq-config-f.pkl'
device = torch.device('cuda')
M=Manipulator()
M.device=device
G=M.LoadModel(network_pkl, device)
M.G=G
M.SetGParameters()
num_img=100_000
M.GenerateS(num_img=num_img)
M.GetCodeMS()
M.alpha=[4.1]


fs3=np.load('../stylegan_model/styleclip/npy/ffhq/fs3.npy')


def run_model(model, img_path, mask_path, save_dir, dif_prompt, gan_prompt):
    if model["type"] is "stylegan":
        print(f"---LOADING MODEL: StyleGAN {model['model_id']} (this may take a little time)---")
        gan_enc = paths_config.model_paths[model['model_id']]
        GAN_net, _ = setup_model(gan_enc,'cuda',encoder_only=False)
        print(f"---FINISHED LOADING MODEL: StyleGAN {model['model_id']}---")
        run_stylegan(model, GAN_net, img_path, save_dir, gan_prompt)
    else: 
        print(f"---LOADING MODEL: {model['model_id']} (this may take a little time)---")
        diffuser_type = getattr(diffusers, model['type'])
        # DIF_model = diffuser_type.from_pretrained(model['model_id'],revision="fp16", torch_dtype=torch.float16).to('cuda')
        # DIF_model = diffuser_type.from_pretrained(model['model_id'], torch_dtype=torch.float16, add_watermarker=False, variant="fp16", use_safetensors=True).to('cuda')

        DIF_model = diffuser_type.from_pretrained(model['model_id'], torch_dtype=torch.float16, use_safetensors=True).to('cuda')
        print(f"---FINISHED LOADING MODEL: {model['model_id']}---")
        run_diffuser(model, DIF_model, img_path, mask_path, save_dir, dif_prompt)



def run_stylegan(model, GAN_net, img_path, save_dir, target):
    restyle = (model['model_id'] == 'restyle')
    img = Image.open(img_path)
    img_name = img_path.name
    print(f"---NOW RUNNING {model['name']} ON: {img_name}---")
    faces_img, _ = align_face(img) 
    neutral='a face' 
    classnames=[target, neutral]
    dt=GetDt(classnames, model_clip)

    for i, face in enumerate(faces_img):
        face.save(save_dir.joinpath(f"cropped_{i}_{model['name']}_{img_name}"), 'PNG')
        #preprocess the image and run the actual attack
        X = preprocess(face,512)
        X = X.to('cuda')

        generator = GAN_net.decoder #init the styleGAN generator
        generator.eval()

        #run the styleGAN generator on the original image and save the result
        X_gan = swap_model(X,'gan')
        latent = get_GAN_latents(GAN_net, X_gan, restyle)   

        imgs_inv, codes = generator([latent], input_is_latent=True, randomize_noise=False, return_latents=True)
        save_image(imgs_inv[0], save_dir, f"result_{i}_{model['name']}_{img_name}")

        # StyleCLIP
        boundary_tmp2, _ = GetBoundary(fs3, dt, M, threshold = 0.15)
        dlatents_loaded=M.G.synthesis.W2S(codes)
        img_indexs=[0]
        dlatents_loaded=M.S2List(dlatents_loaded)
        dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]
        M.num_images=len(img_indexs)
        M.manipulate_layers=None
        codes = M.MSCode(dlatent_tmp, boundary_tmp2)
        out = M.GenerateImg(codes)
        generated=Image.fromarray(out[0,0])
        generated=Image.fromarray(out[0,0])
        generated.save(save_dir.joinpath(f"clip_{i}_{model['name']}_{img_name}"), 'PNG')

        #free GPU space
        del X
        torch.cuda.empty_cache()
        
    


def run_diffuser(model, DIF_model, img_path, mask_path, save_dir, prompt):
    img = Image.open(img_path)
    mask = Image.open(mask_path)

    img_name = img_path.name
    print(f"---NOW RUNNING {model['model_id']} ON: {img_name}---")

    #preprocess the image and run the actual attack
    img = preprocess(img,512).to('cuda')
    mask = preprocess(mask,512).to('cuda')

    # One can fix the seed if needed
    SEED = np.random.randint(low=0, high=10000)


    #these parameters can be tweaked
    # STRENGTH = 0.1
    GUIDANCE = 7.5
    NUM_STEPS = 250

    # with torch.autocast('cuda'):
    #run the diffusion model on the original image and the prompt and save the result
    torch.manual_seed(SEED)
    img = fix_for_diffusion(img)
    mask = fix_for_diffusion(mask)

    img = to_pil(img).convert("RGB")
    mask = to_pil(mask).convert("RGB")

    negative_prompt = "low quality, bad quality, cartoon, 3d, painting, disfigured, b&w"

    if model['strength'] is None:
        image_nat = DIF_model(prompt=prompt, negative_prompt=negative_prompt, image=img, mask_image=mask,
                            num_inference_steps=NUM_STEPS).images[0]
    else:
        image_nat = DIF_model(prompt=prompt, negative_prompt=negative_prompt, image=img, mask_image=mask,
                            strength=model['strength'], num_inference_steps=NUM_STEPS).images[0]

    image_nat.save(save_dir.joinpath(f"{model['name']}_{img_name}"), 'PNG')

    #free GPU space
    del img
    del mask
    torch.cuda.empty_cache()




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
        X = torch.cat([X, GAN_net.get_avg_image()], dim=1)
    codes = GAN_net.encoder(X)
    if GAN_net.opts.start_from_latent_avg:
        if codes.ndim == 2:
            codes = codes + GAN_net.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
        else:
            codes = codes + GAN_net.latent_avg.repeat(codes.shape[0], 1, 1)
    return codes



def save_args(args,save_dir):
    """
    saves the values of args to a file in save_dir
    :param args: the args
    :param save_dir: the save dir
    :return: None
    """
    with open(os.path.join(save_dir,'arguments.txt'),'w') as wrt:
        for arg in vars(args):
            try:
                argument = 'argument: ' + arg + ' is: ' + str(getattr(args,arg)) + '\n'
            except:
                argument = 'argument: ' + arg + ' is: None\n'
            wrt.write(argument)
        wrt.close()


    

def create_dict(name, type, model_id, strength=""):
    return {'name': name, 'type': type, 'model_id': model_id, 'strength': strength}



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main script for running image on varius models")

    parser.add_argument("--im_path", type=str, help='path to the original image')
    parser.add_argument("--mask_path", type=str, help='path to the mask of the image')
    parser.add_argument("--save_dir", type=str, default='./output/result', help='directory name for the result in ./output/')
    parser.add_argument("--dif_prompt", type=str, help='enter a prompt for the diffusers')
    parser.add_argument("--gan_prompt", type=str, help='enter a prompt for Stylegan')


    args = parser.parse_args()
    assert os.path.isfile(args.im_path)

    models = [
        create_dict("SD-v1-5", "StableDiffusionInpaintPipeline", "runwayml/stable-diffusion-v1-5", 0.79),
        create_dict("SD-2-1", "StableDiffusionInpaintPipeline", "stabilityai/stable-diffusion-2-1-base", 0.79),
        create_dict("SD-XL", "StableDiffusionXLInpaintPipeline", "stabilityai/stable-diffusion-xl-base-1.0", 0.82),
        create_dict("openjourney", "StableDiffusionInpaintPipeline", "prompthero/openjourney-v4", 0.75),
        create_dict("dreamlike", "StableDiffusionInpaintPipeline", "dreamlike-art/dreamlike-photoreal-2.0", 0.73),
        create_dict("stylegan-e4e", "stylegan", "e4e"),
        create_dict("stylegan-psp", "stylegan", "psp"),
        create_dict("stylegan-restyle", "stylegan", "restyle"),
        # dall-E - isn't free and closed source 
        # midjourney - doesn't have inpainting 
        # imagen - didn't succesed to get early acsess from google
    ]

    print("---CREATING DIRS---")
    # open all the directories for the results of the attack
    img_path = Path(args.im_path)
    mask_path = Path(args.mask_path) 
    dif_prompt = args.dif_prompt
    gan_prompt = args.gan_prompt

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)

    save_args(args,save_dir)

    print("---FINISHED CREATING DIRS---")
    
    for model in models:
        run_model(model, img_path, mask_path, save_dir, dif_prompt, gan_prompt)
        




