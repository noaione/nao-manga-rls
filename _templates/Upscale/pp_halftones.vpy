from pathlib import Path

import vapoursynth as vs

from upscale_common import (
    bulk_open_images,
    denoise_step,
    get_image_from_subfolders,
    postprocess_image,
    save_frame,
)


def filterchain(clip: vs.VideoNode):
    print("===> Running SMDegrain + BM3D")
    return denoise_step(clip, thSAD=190, sigma=0.78)


CURRENT_DIR = Path(__file__).absolute().parent
TARGET_DIR = CURRENT_DIR / "pp"

print("Preparing...")
all_images = get_image_from_subfolders(CURRENT_DIR / "_temp")
print(f"Got {len(all_images.keys())} volume to process...")

for volume, imaging in all_images.items():
    TARGET_VOL = TARGET_DIR / volume
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
