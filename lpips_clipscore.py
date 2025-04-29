from torchmetrics.multimodal.clip_score import CLIPScore
metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")

import torch
import lpips

loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores


from PIL import Image
from torchvision import transforms

# 定义转换
transform = transforms.PILToTensor()

def path_to_tensor(image_path: str) -> torch.Tensor:
    return transform(Image.open(image_path).resize((512,512))).unsqueeze(0)

def compute_clipscore(target_image_path: str, target_prompt: str) -> float:
    target_image = path_to_tensor(target_image_path)
    return metric(target_image, target_prompt).item()

def compute_lpips(source_image_path: str, target_image_path: str) -> float:
    source_image = path_to_tensor(source_image_path)
    target_image = path_to_tensor(target_image_path)
    return loss_fn_alex(source_image, target_image).item()
