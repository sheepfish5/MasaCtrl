# -- coding: UTF-8

"""
    用来处理 input_dir/meta.json.

    读取 meta.json
    
    更新 meta.json

    写入 meta.json

    所有的数据都直接从 input_dir 和 output_dir/pytorch-tensor 中获取

    meta.json 只用来记录图片描述

"""

from pathlib import Path
from PIL import Image
import json
from dataclasses import asdict, dataclass, is_dataclass, field
from typing import Any, Dict, Literal, Callable, Tuple, List
from tqdm import tqdm

from commons import *

@dataclass
class SingleImage:

    id: int
    season: Season
    prompt: str = " "

@dataclass
class Counter:
    spring: int = 0
    summer: int = 0
    autumn: int = 0
    winter: int = 0

    def add_image(self, image_season: Season):
        if image_season == Season.SPRING:
            self.spring += 1
        elif image_season == Season.SUMMER:
            self.summer += 1
        elif image_season == Season.AUTUMN:
            self.autumn += 1
        elif image_season == Season.WINTER:
            self.winter += 1

@dataclass
class MetaData:
    counter: Counter = field(default_factory=Counter)
    image_data: Dict[str, SingleImage] = field(default_factory=dict)

    def add_image(self, image_path: str, image_id: int, image_season: Season, image_description: str = " "):
        self.image_data[image_path] = SingleImage(image_id, image_season, image_description)
        self.counter.add_image(image_season)
        self.save()

    def save(self):
        _save_metadata_to_json(self, meta_json)

    def traverse_images(self, action: Callable[[str, SingleImage], None]):
        """
            参数:
            ```
                action(image_path: str, single_image: SingleImage)
            ```
        """
        with tqdm(self.image_data.items(), total=len(self.image_data), desc="正在处理图片") as pbar:
            for image_path, image in pbar:
                action(image_path, image)

    def image_num(self) -> int:
        return len(self.image_data)
    
    def check_image_id(self, image_id: int, image_season: Season) -> bool:
        image_path = input_dir / image_season.value / f"{image_id:02d}.jpg"
        return str(image_path) in self.image_data

    def query_single_image(self, image_id: int, image_season: Season) -> SingleImage:
        image_path = input_dir / image_season.value / f"{image_id:02d}.jpg"
        return self.image_data[str(image_path)]
    
    def get_season_image_ids(self, image_season: Season) -> List[int]:
        """返回该季节的所有图片 id"""
        image_ids = []
        for image_path, image in self.image_data.items():
            if image.season == image_season:
                image_ids.append(image.id)
        return image_ids

    @staticmethod
    def load() -> 'MetaData':
        if meta_json.exists():
            return _load_metadata_from_json(meta_json)
        return MetaData()

def _load_metadata_from_json(path: str | Path) -> MetaData:
    with open(path, 'r') as f:
        raw = json.load(f)

    # 1. 解析 counter
    counter_data = raw['counter']
    counter = Counter(**counter_data)

    # 2. 解析 image_data
    image_data_raw = raw['image_data']
    image_data: Dict[str, SingleImage] = {}

    for k, v in image_data_raw.items():
        image = SingleImage(
            id=v['id'],
            season=Season(v['season']),
            prompt=v['prompt'],
        )
        image_data[k] = image

    # 3. 构造最终 MetaData
    return MetaData(counter=counter, image_data=image_data)

def _save_metadata_to_json(meta: MetaData, path: str | Path):
    
    class CustomEncoder(json.JSONEncoder):
        def default(self, o):
            if is_dataclass(o) and not isinstance(o, type):  # 确保 o 是数据类实例而非类型
                return asdict(o)
            elif isinstance(o, Enum):
                return o.value
            return super().default(o)
    
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(meta, f, cls=CustomEncoder, indent=4, ensure_ascii=False)

