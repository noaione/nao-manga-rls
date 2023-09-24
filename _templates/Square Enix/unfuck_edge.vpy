from typing import Literal
import vapoursynth as vs
from vapoursynth import core
import math
from mvsfunc import ToYUV
from vardefunc import region_mask
from vsmasktools import region_rel_mask
from vstools import depth, get_y
from pathlib import Path


def prepare(clip: vs.VideoNode):
    if clip.height % 2 != 0:
        clip = core.resize.Spline36(clip, clip.width + 1, clip.height + 1)
    as_yuv = ToYUV(clip)
    as_yuv = depth(as_yuv, 16)
    # Resize if needed
    return as_yuv


def fix_border(clip: vs.VideoNode, align: Literal["left"] | Literal["right"] = "left"):
    # make adjustment to the first row
    # just apply it to everything for now, we will make the mask later
    ltl = "1.028" if align == "left" else "1.0275"
    cc0 = core.std.Expr(clip, f"x {ltl} *")
	# make adjustment to the second row
    rtl = "1.01" if align == "left" else "1.0"
    cc1 = core.std.Expr(clip, f"x {rtl} *")

    clip_white = core.std.BlankClip(clip, color=[255, 255, 255])

    white_line = core.std.Crop(get_y(clip_white), 0, clip.width - 1, 0, 0).std.Invert()
	# white line on black bg
    ccm0 = core.std.AddBorders(white_line, 0, clip.width - 1, 0, 0, 0)
    ccm1 = core.std.AddBorders(white_line, 1, clip.width - 2, 0, 0, 0)

    if align == "right":  # rotate if we want to apply to the right side
        ccm0 = core.vcm.Rotate(ccm0, ccm0, 180)
        ccm1 = core.vcm.Rotate(ccm1, ccm1, 180)

    # apply mask
    c0 = core.std.MaskedMerge(clip, cc0, ccm0)
    c1 = core.std.MaskedMerge(c0, cc1, ccm1)

    return c1


vol_num = "v12"

cdir = Path.cwd() / "pp" / vol_num
img_col: vs.VideoNode = None
for im in cdir.glob("*.png"):
    _, pnum = im.stem.rsplit("_", 1)
    if "cover" in im.stem:
        pnum = "0"
    num = int(pnum)
    if num < 3:
        continue
    img = core.imwri.Read(str(im))
    or_w, or_h = img.width, img.height
    img = prepare(img)
    img = core.resize.Spline36(img, or_w, or_h)
    align = "left" if num % 2 == 0 else "right"
    img = fix_border(img, align)
    img = get_y(img)
    if img_col is None:
        img_col = img
    else:
        img_col += img


(Path.cwd() / "out" / vol_num).mkdir(parents=True, exist_ok=True)
final_output = f"out/{vol_num}/i-markfs_{vol_num[1:]}_" + "%03d.png"
time_to_write = core.imwri.Write(depth(img_col, 8), "PNG", final_output, 3, 100)
time_to_write.set_output(0)

if __name__ == "__main__":
    for i in range(time_to_write.num_frames):
        print(f"Writing frame {i}")
        time_to_write.get_frame(i)
