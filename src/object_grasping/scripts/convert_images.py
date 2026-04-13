import os
from pathlib import Path
from PIL import Image

import torch
from torchvision.transforms import v2

# -----------------------
# CONFIG
# -----------------------
DIR = "/home/joeya/dataset/demo1"

# -----------------------
# TRANSFORM: PIL -> uint8 tensor (C, H, W)
# -----------------------
to_tensor_uint8 = v2.Compose([
    v2.ToImage(),
    v2.Lambda(lambda x: (x * 255.0).to(torch.uint8)),
    v2.Resize(256),
    v2.CenterCrop(224),
])


# -----------------------
# MAIN CONVERSION
# -----------------------
def convert_images():
    img_paths = list(Path(DIR).rglob("*.jpg"))

    print(f"Found {len(img_paths)} images")

    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = to_tensor_uint8(img)

            torch.save(tensor, img_path.with_suffix(".pt"))

        except Exception as e:
            print(f"Failed on {img_path}: {e}")

if __name__ == "__main__":
    convert_images()