from run_image_edit import generate_image
from json_process import *

output_dir = Path("output_dir")
output_dir.mkdir(parents=True, exist_ok=True)

# 直接运行测试
image_data_list = {
    "spring": [i for i in range(8, 53+1)],
    "summer": [i for i in range(7, 53+1)],
    "autumn": [i for i in range(5, 52+1)],
    "winter": [i for i in range(13, 54+1)],
}

def action(image_path: str, single_image: SingleImage):

    global image_data_list

    if single_image.id not in image_data_list[single_image.season.value]:
        return
    
    generate_image(image_path, single_image, output_dir)

if __name__ == "__main__":
    meta_data = MetaData.load()
    # 遍历所有图片
    meta_data.traverse_images(action)