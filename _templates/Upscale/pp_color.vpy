from pathlib import Path

import havsfunc as hvf
import vapoursynth as vs

from upscale_common import (
    bulk_open_images,
    denoise_step,
    get_image_from_subfolders,
    postprocess_image,
    save_frame,
)
from vapoursynth import core


def filterchain(clip: vs.VideoNode):
    print("===> Applying Dehalo")
    dehalo = hvf.FineDehalo(clip, rx=2, thlimi=80, thlima=120, contra=0.5)
    print("===> Applying DAA")
    # Check if clip is mod 2
    if clip.height % 2 != 0:
        # If not, resize to the nearest mod 2
        dehalo = core.resize.Spline36(dehalo, clip.width + 1, clip.height + 1)
    aa = hvf.daa(dehalo, nsize=4, nns=2, qual=2, opencl=True, device=0)
    if clip.height % 2 != 0:
        # If not, resize to the nearest mod 2
        aa = core.resize.Spline36(aa, clip.width, clip.height)
    print("===> Applying Post-AA Dehalo")
    postaa = hvf.FineDehalo(aa)
    print("===> Applying SMDegrain")
    return denoise_step(postaa, thSAD=200, two_step=False)
    # print("===> Applying SMDegrain + BM3D")
    # return denoise_step(postaa, thSAD=200, two_step=True)


CURRENT_DIR = Path(__file__).absolute().parent
# TARGET_DIR = CURRENT_DIR / "postprocess"
TARGET_DIR = CURRENT_DIR / "pp"

print("Preparing...")
all_images = get_image_from_subfolders(CURRENT_DIR / "_pgc")
# all_images = get_image_from_subfolders(CURRENT_DIR / "_tonedfix")
print(f"Got {len(all_images.keys())} volume to process...")

for volume, imaging in all_images.items():
    TARGET_VOL = TARGET_DIR / f"{volume}_pgc"
    TARGET_VOL.mkdir(exist_ok=True)
    print(f"=> Processing {volume}...")

    print(f"==> Opening images as clip... ({len(imaging)} images)")
    img_xclips, img_xpairs = bulk_open_images(imaging)
    print(f"==> Got {len(img_xclips)} chunks!")
    for ci, img_chunk in enumerate(img_xclips, 1):
        img_pairs = img_xpairs[ci - 1]
        print(f"==> Running filterchain (chunk {ci})...")
        img_chunk = filterchain(img_chunk)
        print(f"==> Post-processing images (chunk {ci})...")
        img_chunk = postprocess_image(img_chunk)

        print(f"==> Saving chunk {ci} for {volume}")
        for fidx in range(img_chunk.num_frames):
            fr_c = img_chunk.get_frame(fidx)
            fr_p = img_pairs[fidx]
            new_path = TARGET_VOL / fr_p.filename
            print(f"===> Saving frame {fr_p.filename}")
            save_frame(fr_c, new_path)
            del fr_c  # dealloc
        del img_chunk  # dealloc
