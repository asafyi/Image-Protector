# Image Protector

## Getting Started

We created an interactive website that allows users to upload their images and get them immunized from AI image manipulation. 

Notice that because of size limitation, we couldn’t contain a few heavy files (StyleGan's ffhq encode and react modules) so not all things will be able to run.  

The missing files are:
- /main/web/node_modules/ - only needed if want to rebuild the site from react
- /stylegan_model/styleclip/model/stylegan2-ffhq-config-f.pkl - needed for image_testing/
- /stylegan_model/weights/e4e_ffhq_encode.pt - needed for image_testing/ and attacks/
- /stylegan_model/weights/psp_ffhq_encode.pt - needed for image_testing/ and attacks/
- /stylegan_model/weights/restyle_e4e_ffhq_encode.pt - needed for image_testing/ and attacks/
    
You can complete these files from web in order to test it.


In order to run these programs locally, you will need Cuda on you machine and these Python dependecies:
    - torch
    - torchvision
    - numpy
    - matplotlib
    - diffusers["torch"]
    - pillow
    - tqdm
    - scipy
    - dlib
    - cv2
    - flask
    - flask_cors
    - requests
    - Ninja
    - CLIP - from https://github.com/openai/CLIP.git
    - transformers
    - accelerate
    - gunicorn

### Attack Script

The Python script allowed us to test and design our approach for immunizing an image. The script gets various parameters and an image. The script creates a folder containing the immunized image and the results of running the StyleGan model and stable-diffusion model on the original image and the immunized image.

The project has a folder called "attacks" where the script is stored, and and here is a sample run:

```bash
python3 main.py --epsilon 0.04 --pgd_iters 100  --im_dir ./bill_images --save_dir ./bill_attack5 --prompt "a man in the park"
```

The parameters allow the user to choose various StyleGan / stable-diffusion models, different approaches to calculate loss, enter a prompt, choose the amount of maximum noise in the image, the number of iterations, etc.

You can see all the options with explanations by running:

```bash
python3 main.py --help 
```

### Verification Script

The Python script allowed us to test for the report our approach for immunizing an image in various StyleGan and Diffusion models. Given an image (immunized or not), mask, directory, and a prompt, the script runs image manipulation using various StyleGan and Diffusion models and saves the result in a directory. 

The project has a folder called “image_testing” where the script is stored, and here is a sample run:

```bash
python3 main.py --im_path ./input/taylor.jpg --mask_path ./input/taylor_mask2.jpg  --save_dir ./output/taylor_mall/ --dif_prompt "a woman in a mall" --gan_prompt "a face with black hair"
```

### Web Server

The web server implementation is located under the folder “main” and contains:
- web/ - contains all of the original code for the website (React and Typescript)
- dist/ - contains all the client-side code of the website (built from the code in the 'web' folder)
- server.py - the website's server (using flask)
- server_redirect.py - a flask server only for redirecting every HTTP request to HTTPS
- attack.py - contains the actual attack method that the server runs.
- gunicorn_config.py - config file for gunicorn to run the website.

To run the website:
```bash
gunicorn --config gunicorn_config.py server:app --timeout 2000 --threads 4
gunicorn --bind 0.0.0.0:80 server_redirect:app
```   
### Utils

The folder “image_utils” contains a few modules that other scripts use for face recognition, image processing, etc.

***Made by [@asafyi](https://github.com/asafyi) && [@galgodsi](https://github.com/galgodsi) && [@yotammaoz](https://github.com/yotammoaz)***
