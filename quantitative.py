from json_process import *

from torchvision import transforms
from PIL import Image
from pathlib import Path

from lpips_clipscore import compute_clipscore, compute_lpips

@dataclass
class SingleImageData:
    lpips: float
    clip_score: float

total_data: Dict[str, SingleImageData] = {}

def add_data(source_image: SingleImage, target_season: Season, target_image_data: SingleImageData):
    """向 total_data 添加一条数据"""
    target_image_key = f"{source_image.season.value}-{source_image.id:02}-to-{target_season.value}"
    total_data[target_image_key] = target_image_data

def save_data():
    """保存 total_data 到 quantitative.json"""
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if is_dataclass(o) and not isinstance(o, type):  # 确保 o 是数据类实例而非类型
                return asdict(o)
            elif isinstance(o, Enum):
                return o.value
            return super().default(o)
    
    with open("quantitative.json", 'w', encoding="utf-8") as f:
        json.dump(total_data, f, cls=CustomEncoder, indent=4, ensure_ascii=False)

def search_target_image_path_by(source_image: SingleImage, target_season: Season) -> str:
    result = f"output_dir/{source_image.season.value}-{source_image.id:02d}-to-{target_season.value}.jpg"
    return result


counter = 0
total_lpips = 0.0
total_clipscore = 0.0
# average_lpips = 0.24209230341787996
# average_clipscore = 33.08898840279415

# total
# average_lpips = 0.24633388475473947
# average_clipscore = 31.544177589176584

SEASONS = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
def action(image_path: str, single_image: SingleImage):

    global counter, total_lpips, total_clipscore

    for target_season in SEASONS:
        if target_season == single_image.season: continue

        target_image_path = search_target_image_path_by(single_image, target_season)
        assert target_image_path != ""
        target_prompt = f"{single_image.prompt} at {target_season.value}"
        clipscore = compute_clipscore(target_image_path, target_prompt)
        lpips = compute_lpips(image_path, target_image_path)
        target_image_data = SingleImageData(lpips, clipscore)

        total_lpips += lpips
        total_clipscore += clipscore
        counter += 1
        
        add_data(single_image, target_season, target_image_data)
        save_data()


if __name__ == '__main__':
    meta_data = MetaData.load()
    meta_data.traverse_images(action)

    average_lpips = total_lpips / counter
    average_clipscore = total_clipscore / counter
    print(f"average lpips: {average_lpips}")
    print(f"average clipscore: {average_clipscore}")