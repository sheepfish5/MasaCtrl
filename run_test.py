import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from tqdm import tqdm
from einops import rearrange, repeat
from omegaconf import OmegaConf

from diffusers import DDIMScheduler

from masactrl.diffuser_utils import MasaCtrlPipeline
from masactrl.masactrl_utils import AttentionBase
from masactrl.masactrl_utils import regiter_attention_editor_diffusers

from torchvision.utils import save_image
from torchvision.io import read_image
from pytorch_lightning import seed_everything

from pathlib import Path

torch.cuda.set_device(0)  # set the GPU device

# Note that you may add your Hugging Face token to get access to the models
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model_path = "xyn-ai/anything-v4.0"
model_path = "CompVis/stable-diffusion-v1-4"
# model_path = "runwayml/stable-diffusion-v1-5"
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

local_model_path = Path("./stable-diffusion-v1-4")
if local_model_path.exists():
    model = MasaCtrlPipeline.from_pretrained(local_model_path, scheduler=scheduler).to(device)
else:
    model = MasaCtrlPipeline.from_pretrained(model_path, scheduler=scheduler).to(device)
    model.save_pretrained(local_model_path)



from masactrl.masactrl import MutualSelfAttentionControl
from torchvision.io import read_image


def load_image(image_path, device):
    """读入图片，插值到 512x512 """
    image = read_image(image_path)
    image = image[:3].unsqueeze_(0).float() / 127.5 - 1.  # [-1, 1]
    image = F.interpolate(image, (512, 512))
    image = image.to(device)
    return image


seed = 42
seed_everything(seed)

out_dir = "./workdir/masactrl_real_exp/"
os.makedirs(out_dir, exist_ok=True)
sample_count = len(os.listdir(out_dir))
out_dir = os.path.join(out_dir, f"sample_{sample_count}")
os.makedirs(out_dir, exist_ok=True)

# source image
SOURCE_IMAGE_PATH = "./gradio_app/images/corgi.jpg"
source_image = load_image(SOURCE_IMAGE_PATH, device)

source_prompt = ""
target_prompt = "a photo of a running corgi"
prompts = [source_prompt, target_prompt]

# invert the source image
start_code, latents_list = model.invert(source_image,
                                        source_prompt,
                                        guidance_scale=7.5,
                                        num_inference_steps=50,
                                        return_intermediates=True)
start_code = start_code.expand(len(prompts), -1, -1, -1)

# results of direct synthesis
editor = AttentionBase()
regiter_attention_editor_diffusers(model, editor)
image_fixed = model([target_prompt],
                    latents=start_code[-1:],
                    num_inference_steps=50,
                    guidance_scale=7.5)

save_image(image_fixed, os.path.join(out_dir, f"fixed.png"))

print("Syntheiszed images are saved in", out_dir)