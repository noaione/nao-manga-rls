from re import L
import vapoursynth as vs
from vapoursynth import core
from mvsfunc import ToYUV, ToRGB
from havsfunc import SMDegrain
from vstools import depth, join, split
from pathlib import Path

ROOT_DIR = Path(__file__).absolute().parent
DENOISE_TARGET = ROOT_DIR / "denoised"


def prepare(clip: vs.VideoNode):
    if clip.height % 2 != 0:
        clip = core.resize.Spline36(clip, clip.width + 1, clip.height + 1)
    as_yuv = ToYUV(clip)
    as_yuv = depth(as_yuv, 16)
    # Resize if needed
    return as_yuv


def postpare(clip: vs.VideoNode):
    or_w, or_h = clip.width, clip.height
    if clip.height % 2 != 0:
        clip = core.resize.Spline36(clip, clip.width + 1, clip.height + 1)
    as_rgb = ToRGB(clip)
    as_rgb = depth(as_rgb, 8)
    if clip.height % 2 != 0:
        as_rgb = core.resize.Spline36(as_rgb, or_w, or_h)
    return as_rgb


def get_number(page_stem: str):
    ss = "_" if "_" in page_stem else " "
    _, pnum = page_stem.rsplit(ss, 1)
    num = int(pnum.strip("p"))
    return num


def process_filterchain(folder_path: Path):
    img_collate: vs.VideoNode = None
    actual_filename: list[str] = []
    for im in folder_path.glob("*.png"):
        number = int(im.stem[1:])
        parent_ch = im.parent.name
        _test_path = DENOISE_TARGET / parent_ch / f"p{(number - 1):03d}.png"
        if _test_path.exists():
            print(f"  Skipping {im.stem}")
            continue
        clip = core.imwri.Read(str(im))
        or_w, or_h = clip.width, clip.height
        clip = prepare(clip)
        clip = core.resize.Spline36(clip, or_w, or_h)
        if img_collate and clip.height != img_collate.height:
            print(f"Different height detected!, skipping: {im.name}")
        if img_collate is None:
            img_collate = clip
        else:
            img_collate += clip
        actual_filename.append(im.stem)
    if img_collate is None:
        print("No images to process, skipping")
        vs.clear_outputs()
        return

    split_planes = split(img_collate)
    if len(split_planes) == 1:
        luma = img_collate
        chroma = None
    else:
        print("Splitting image for optimization")
        luma = split_planes[0]
        chroma = split_planes[1:]

    filt_dny_ref = SMDegrain(luma, tr=2, thSAD=180, thSADC=0, RefineMotion=True, prefilter=2)
    filt_dny = core.bm3dcuda.BM3D(
        depth(luma, 32), ref=depth(filt_dny_ref, 32), sigma=0.88, radius=2
    ).bm3d.VAggregate(radius=2)

    if chroma is not None:
        print("Recombining planes")
        filt_dny = join([filt_dny, *chroma])

    ch_name = folder_path.name
    out_dir = DENOISE_TARGET / ch_name / "_ttemp%03d.png"
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_clip = core.imwri.Write(postpare(filt_dny), "PNG", str(out_dir), 0, 100)
    out_clip.set_output(0)

    print("Waiting for process to finish...")
    for i in range(out_clip.num_frames):
        targ_name = actual_filename[i]
        if targ_name.startswith("p"):
            targ_name = targ_name[1:]
        targ_name = int(targ_name)
        denoise_fn = DENOISE_TARGET / ch_name / f"p{targ_name:03d}.png"
        source_fn = DENOISE_TARGET / ch_name / f"_ttemp{i:03d}.png"
        print(f"  Outputting {source_fn}")
        out_clip.get_frame(i)
        print(f"  Renaming {source_fn} to {denoise_fn}")
        source_fn.rename(denoise_fn)
    # Cleanup
    vs.clear_outputs()


if __name__ == "__main__":
    OUT_DIR = ROOT_DIR / "out"
    for folder in OUT_DIR.glob("*"):
        if folder.is_dir() and folder.name.startswith("c"):
            print(f"Processing {folder.name}")
            process_filterchain(folder)
