from pathlib import Path

import imagequant
from PIL import Image, ImageEnhance

from upscale_common import get_image_from_subfolders

CURRENT_DIR = Path(__file__).absolute().parent
TARGET_DIR = CURRENT_DIR / "prefinal"

print("Preparing...")
all_images = get_image_from_subfolders(CURRENT_DIR / "pp")
print(f"Got {len(all_images.keys())} volume to process...")

for volume, imaging in all_images.items():
    TARGET_VOL = TARGET_DIR / volume
    TARGET_VOL.mkdir(exist_ok=True)
    print(f"=> Processing {volume}...")

    for image_path in imaging:
        print(f"==> Processing {image_path.path.name}...")
        im = Image.open(image_path.path)
        quant = imagequant.quantize_pil_image(
            im, dithering_level=1.0, max_colors=16
        )
        # Black level is at #010101, contrast stretch to #000000
        quant = ImageEnhance.Contrast(
            quant.convert("RGB")
        ).enhance(1.008).convert("L")
        quant.save(
            fp=TARGET_VOL / image_path.path.name,
            format="png",
            optimize=False
        )
