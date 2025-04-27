from run_image_edit import generate_image
from json_process import *

output_dir = Path("output_dir")
output_dir.mkdir(parents=True, exist_ok=True)

def action(image_path: str, single_image: SingleImage):
    seasons = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
    for target_season in tqdm(seasons, total=4, desc="正在生成四季图片"):
        source_image_path = image_path
        target_prompt = f"{single_image.prompt} at {target_season.value}"
        output_image_path = output_dir / f"{single_image.season}-{single_image.id}-to-{target_season.value}.jpg"

        generate_image(source_image_path, target_prompt, output_image_path)

if __name__ == "__main__":
    meta_data = MetaData.load()
    # 遍历所有图片
    meta_data.traverse_images(action)