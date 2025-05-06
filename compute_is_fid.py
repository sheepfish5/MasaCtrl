from json_process import *
import torch
from torchvision import transforms
from PIL import Image

transform = transforms.PILToTensor()

def search_target_image_path_by(source_image: SingleImage, target_season: Season) -> str:
    result = f"output_dir/{source_image.season.value}-{source_image.id:02d}-to-{target_season.value}.jpg"
    return result


def path_to_tensor(image_path: str) -> torch.Tensor:
    """
        返回值：
            torch.Tensor(1, 3, 299, 299)
    """
    return transform(Image.open(image_path).resize((299,299))).unsqueeze(0)

from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

total_fid = FrechetInceptionDistance(feature=64)
total_is = InceptionScore()

total_sw_fid = FrechetInceptionDistance(feature=64)
total_sw_is = InceptionScore()

SEASONS = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
def action(image_path: str, single_image: SingleImage):

    source_image_tensor = path_to_tensor(image_path)
    total_fid.update(source_image_tensor, real=True)

    if single_image.season in [Season.SUMMER, Season.WINTER]:
        total_sw_fid.update(source_image_tensor, real=True)

    for target_season in SEASONS:
        if target_season == single_image.season: continue

        target_image_path = search_target_image_path_by(single_image, target_season)
        assert target_image_path != ""
        
        image_tensor = path_to_tensor(target_image_path)
        total_fid.update(image_tensor, real=False)
        total_is.update(image_tensor)

        if single_image.season == Season.SUMMER and target_season == Season.WINTER or single_image.season == Season.WINTER and target_season == Season.SUMMER:
            total_sw_fid.update(image_tensor, real=False)
            total_sw_is.update(image_tensor)


if __name__ == '__main__':
    meta_data = MetaData.load()
    meta_data.traverse_images(action)

    avg_fid = total_fid.compute()
    is_mean, is_std = total_is.compute()

    avg_sw_fid = total_sw_fid.compute()
    sw_is_mean, sw_is_std = total_sw_is.compute()

    with open("./fid_is.json", "w") as f:
        json.dump({
            "fid": avg_fid.item(),
            "is_mean": is_mean.item(),
            "is_std": is_std.item(),
            "sw_fid": avg_sw_fid.item(),
            "sw_is_mean": sw_is_mean.item(),
            "sw_is_std": sw_is_std.item()
        }, f, indent=4)
    
    print(f"avg_fid: {avg_fid.item()}")
    print(f"is_mean: {is_mean.item()}")
    print(f"is_std: {is_std.item()}")
    print(f"avg_sw_fid: {avg_sw_fid.item()}")
    print(f"sw_is_mean: {sw_is_mean.item()}")
    print(f"sw_is_std: {sw_is_std.item()}")