from run_image_edit import generate_image
from json_process import *

output_dir = Path("output_dir")
output_dir.mkdir(parents=True, exist_ok=True)

def action(image_path: str, single_image: SingleImage):
    
    generate_image(image_path, single_image, output_dir)

if __name__ == "__main__":
    meta_data = MetaData.load()
    # 遍历所有图片
    meta_data.traverse_images(action)