import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import v2

to_tensor_uint8 = v2.Compose(
    [
        v2.ToImage(),
        v2.Lambda(lambda x: (x * 255.0).to(torch.uint8)),
        v2.Resize(256),
        v2.CenterCrop(224),
    ]
)


def convert_images(directory: str):
    """Converts jpg images in a directory to pytorch tensors."""

    img_paths = list(Path(directory).rglob("*.jpg"))

    print(f"Found {len(img_paths)} images")

    for img_path in img_paths:
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = to_tensor_uint8(img)

            torch.save(tensor, img_path.with_suffix(".pt"))

        except Exception as e:
            print(f"Failed on {img_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("directory")
    args = parser.parse_args()

    convert_images(parser.directory)
