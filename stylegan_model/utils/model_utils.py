import torch
import argparse
from models.psp import pSp
from models.restyle import REStyle
from models.encoders.psp_encoders import Encoder4Editing




def setup_model(checkpoint_path, device='cuda', encoder_only=False):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    
    if 'channel_multiplier' not in opts.keys():
        opts['channel_multiplier'] = 2

    is_cars = 'car' in opts['dataset_type']
    is_faces = 'ffhq' in opts['dataset_type']
    if 'stylegan_size' not in opts.keys():
        if is_faces:
            opts['stylegan_size'] = 1024
        elif is_cars:
            opts['stylegan_size'] = 512
        else:
            opts['stylegan_size'] = 256

    opts['checkpoint_path'] = checkpoint_path
    opts['device'] = device
    opts = argparse.Namespace(**opts)
    if 'restyle' in checkpoint_path:
        net = REStyle(opts)
    else:
        net = pSp(opts)
    net.eval()
    net = net.to(device)
    if net._get_name() == 'REStyle':
        net.get_avg_image()
    if encoder_only:
        net.decoder = net.decoder.to(torch.device('cpu'))
        del net.decoder
    return net, opts


def load_e4e_standalone(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    opts = argparse.Namespace(**ckpt['opts'])
    e4e = Encoder4Editing(50, 'ir_se', opts)
    e4e_dict = {k.replace('encoder.', ''): v for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
    e4e.load_state_dict(e4e_dict)
    e4e.eval()
    e4e = e4e.to(device)
    latent_avg = ckpt['latent_avg'].to(device)

    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)

    e4e.register_forward_hook(add_latent_avg)
    return e4e
