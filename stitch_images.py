from PIL import Image, ImageDraw
from json_process import *

def compose_images(image_paths, output_path: str, highlight_index=None,):
    """
    将4张图横向拼接，每张图之间10px间距，整体四周10px边距。
    可选指定某一张图用5px红框高亮（红框占用间距空间，不扩大图像区域）。
    """
    assert len(image_paths) == 4, "需要提供4张图片"
    size = (256, 256)
    spacing = 10
    margin = 10
    frame_width = 5

    processed_images = []

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGB").resize(size)

        if i == highlight_index:
            # 加红框（红色背景，图像居中）
            img_with_border = Image.new("RGB", (size[0] + 2 * frame_width, size[1] + 2 * frame_width), "red")
            img_with_border.paste(img, (frame_width, frame_width))
            # 裁剪中心区域为原始大小，确保尺寸不变
            left = frame_width
            upper = frame_width
            right = left + size[0]
            lower = upper + size[1]
            img = img_with_border.crop((left - frame_width, upper - frame_width, right + frame_width, lower + frame_width))
            # 裁剪后尺寸仍为256x256，但红框“压住”周围像素
        processed_images.append(img)

    total_width = 4 * size[0] + 3 * spacing + 2 * margin
    total_height = size[1] + 2 * margin
    canvas = Image.new("RGB", (total_width, total_height), "white")

    x = margin
    for i, img in enumerate(processed_images):
        if i == highlight_index:
            canvas.paste(img, (x - frame_width, margin - frame_width))
        else:
            canvas.paste(img, (x, margin))
        x += size[0] + spacing

    canvas.save(output_path)

def search_target_image_path_by(source_image: SingleImage, target_season: Season) -> str:
    result = f"output_dir/{source_image.season.value}-{source_image.id:02d}-to-{target_season.value}.jpg"
    return result


SEASONS = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
def action(image_path: str, single_image: SingleImage):

    image_paths: List[str] = []

    source_image_idx = None

    for i, target_season in enumerate(SEASONS):
        if target_season == single_image.season: 
            image_paths.append(image_path)
            source_image_idx = i
        else:
            target_image_path = search_target_image_path_by(single_image, target_season)
            assert target_image_path != ""
            image_paths.append(target_image_path)
    
    output_path = output_compose_dir / f"{single_image.season.value}-{single_image.id:02d}.jpg"

    compose_images(image_paths, str(output_path), source_image_idx)
        


if __name__ == '__main__':
    meta_data = MetaData.load()
    meta_data.traverse_images(action)
