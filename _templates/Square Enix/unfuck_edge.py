import sys
from pathlib import Path

from PIL import Image, ImageOps


CURRENT_DIR = Path(__file__).absolute().parent

try:
    volume_dir = sys.argv[1]
except IndexError:
    print("Usage: python unfuck_edge.py <volume>")
    sys.exit(1)


source_dir = CURRENT_DIR / "pp" / volume_dir
target_dir = CURRENT_DIR / "out" / volume_dir

if not source_dir.exists():
    print("Invalid source dir, does not exist!")
    sys.exit(1)


target_dir.mkdir(exist_ok=True)

MAX_WIDTH_MUL = 1.032
MAX_WIDTH_MUL_RIGHT = 1.035
AFTER_THAT_MUL = 1.01


def do_pixel_justice(im_gray: Image.Image):
    im_maxp = im_gray.point(lambda x: x * MAX_WIDTH_MUL)
    im_maxp_right = im_gray.point(lambda x: x * MAX_WIDTH_MUL_RIGHT)
    im_afterp = im_gray.point(lambda x: x * AFTER_THAT_MUL)

    # For im_maxp, do it on the left and right most pixel row only
    target_im = im_gray.copy()
    for row in range(im_gray.height):
        max_pp = im_maxp.getpixel((0, row))
        after_pp = im_afterp.getpixel((1, row))
        maxmax_pp = im_maxp_right.getpixel((im_gray.width - 1, row))
        # aftermax_pp = im_afterp.getpixel((im_gray.width - 2, row))

        target_im.putpixel((0, row), max_pp)
        target_im.putpixel((1, row), after_pp)
        target_im.putpixel((im_gray.width - 1, row), maxmax_pp)
        # target_im.putpixel((im_gray.width - 2, row), aftermax_pp)
    im_maxp.close()
    im_afterp.close()
    return target_im


cc = 1
for image in source_dir.glob("*.png"):
    print(f"Processing: {image.name}")
    im = Image.open(image)
    im_gray = ImageOps.grayscale(im)

    im_done = do_pixel_justice(im_gray)
    im_gray.close()
    im.close()
    print(f"  Saving: {image.name}")
    im_done.save(target_dir / image.name, "PNG")
