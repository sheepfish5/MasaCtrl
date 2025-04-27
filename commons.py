from enum import Enum, unique
from pathlib import Path

@unique  # 确保所有值都是唯一的
class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    AUTUMN = "autumn"
    WINTER = "winter"

output_dir = Path("output_dir")
input_dir = Path("input_dir")
meta_json = input_dir / Path("meta.json")
output_pytorch_tensor_dir = output_dir / Path("pytorch-tensor")

output_dir.mkdir(parents=True, exist_ok=True)
input_dir.mkdir(parents=True, exist_ok=True)
output_pytorch_tensor_dir.mkdir(parents=True, exist_ok=True)