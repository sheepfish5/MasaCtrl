from json_process import *
import torch
import os

from dataclasses import asdict, dataclass, is_dataclass
from collections import defaultdict
from statistics import mean

@dataclass
class ImageData:
    id: str
    source_season: Season
    target_season: Season
    image_reward: float

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o) and not isinstance(o, type):  # 确保 o 是数据类实例而非类型
            return asdict(o)
        elif isinstance(o, Enum):
            return o.value
        return super().default(o)


# 直接运行测试
image_data_list = {
    "spring": [i for i in range(8, 53+1)],
    "summer": [i for i in range(7, 53+1)],
    "autumn": [i for i in range(5, 52+1)],
    "winter": [i for i in range(13, 54+1)],
}


def search_target_image_path_by(source_image: SingleImage, target_season: Season) -> str:
    result = f"output_dir/{source_image.season.value}-{source_image.id:02d}-to-{target_season.value}.jpg"
    return result



import ImageReward as RM
model = RM.load("ImageReward-v1.0")

global_data: List[ImageData] = []

def save_data():
    with open("quantitative_image_reward.json", "w", encoding="utf-8") as f:
        json.dump(global_data, f, cls=CustomEncoder, indent=4, ensure_ascii=False)

SEASONS = [Season.SPRING, Season.SUMMER, Season.AUTUMN, Season.WINTER]
def action(image_path: str, single_image: SingleImage):

    global global_data, model

    for target_season in SEASONS:
        if target_season == single_image.season: continue

        target_prompt = f"{single_image.prompt} at {target_season.value}"
        target_image_path = search_target_image_path_by(single_image, target_season)
        assert target_image_path != ""
        
        score = model.score(target_prompt, target_image_path)

        image_data = ImageData(
            id=f"{single_image.id:02}",
            source_season=single_image.season,
            target_season=target_season,
            image_reward=score
        )

        global_data.append(image_data)


if __name__ == '__main__':
    meta_data = MetaData.load()
    meta_data.traverse_images(action)

    save_data()

    # 使用 defaultdict 存储每个 (source_season, target_season) 对应的 lpips 和 clip_score
    # (source_season, target_season) -> (lpips_list[], clip_score_list[])
    grouped_data = defaultdict(lambda: {"image_reward": [],})

    # 分组数据
    for item in global_data:
        key = (item.source_season, item.target_season)
        grouped_data[key]["image_reward"].append(item.image_reward)

    # 计算每组的均值
    averages = []
    for key, values in grouped_data.items():
        avg_image_reward = mean(values["image_reward"])
        averages.append({"source_season": key[0], "target_season": key[1], "avg_image_reward": avg_image_reward})

    with open("quantitative_image_reward_statistics.json", "w", encoding="utf-8") as f:
        json.dump(averages, f, cls=CustomEncoder, indent=4)

    # 总共的均值
    total_avg_image_reward = mean([x.image_reward for x in global_data])
    print(f"Total average image reward: {total_avg_image_reward}")
    # Total average image reward: 0.36398560453216183