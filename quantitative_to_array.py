
import json
from pathlib import Path

from commons import *

import re
from typing import Tuple, List

from dataclasses import asdict, dataclass, is_dataclass
from collections import defaultdict
from statistics import mean

@dataclass
class ImageData:
    id: str
    source_season: Season
    target_season: Season
    lpips: float
    clip_score: float

def parse_season_string(s: str) -> Tuple[str, str, str]:
    """
        返回值:
        (source_season, id, target_season)
    """
    pattern = r'^(spring|summer|autumn|winter)-(\d{2})-to-(spring|summer|autumn|winter)$'
    match = re.match(pattern, s)
    # 如果你确定输入一定匹配，可以放心使用 assert
    assert match is not None, "Input string format must be guaranteed to match"
    return match.group(1), match.group(2), match.group(3)

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if is_dataclass(o) and not isinstance(o, type):  # 确保 o 是数据类实例而非类型
            return asdict(o)
        elif isinstance(o, Enum):
            return o.value
        return super().default(o)

if __name__ == '__main__':
    with open("quantitative.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
    
    data: List[ImageData] = []

    # assert type(raw) == dict[str, dict]
    for key, value in raw.items():
        source_season, image_id, target_season = parse_season_string(key)
        image_data = ImageData(
            id=image_id,
            source_season=Season(source_season),
            target_season=Season(target_season),
            lpips=value['lpips'],
            clip_score=value['clip_score'],
        )
        data.append(image_data)
    
    with open("quantitative_to_array.json", "w", encoding="utf-8") as f:

        json.dump(data, f, cls=CustomEncoder, indent=4)

    # 使用 defaultdict 存储每个 (source_season, target_season) 对应的 lpips 和 clip_score
    # (source_season, target_season) -> (lpips_list[], clip_score_list[])
    grouped_data = defaultdict(lambda: {"lpips": [], "clip_score": []})

    # 分组数据
    for item in data:
        key = (item.source_season, item.target_season)
        grouped_data[key]["lpips"].append(item.lpips)
        grouped_data[key]["clip_score"].append(item.clip_score)

    # 计算每组的均值
    averages = []
    for key, values in grouped_data.items():
        avg_lpips = mean(values["lpips"])
        avg_clip_score = mean(values["clip_score"])
        averages.append({"source_season": key[0], "target_season": key[1], "avg_lpips": avg_lpips, "avg_clip_score": avg_clip_score})
        # averages[key] = (avg_lpips, avg_clip_score)

    with open("quantitative_statistics.json", "w", encoding="utf-8") as f:
        json.dump(averages, f, cls=CustomEncoder, indent=4)