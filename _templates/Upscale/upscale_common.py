from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs
from havsfunc import SMDegrain
from mvsfunc import ToRGB, ToYUV
from PIL import Image
from vapoursynth import core
from vstools import depth, join, split
from numpy.typing import NDArray


def save_frame(frame: vs.VideoFrame, save_path: Path):
    f_array: NDArray[Any] = np.dstack(frame)  # type: ignore[call-overload]
    img = Image.fromarray(f_array, "RGB")
    img.save(save_path.with_suffix(".png"), format="PNG", optimize=False, compress_level=9)


@dataclass
class ImageData:
    filename: str
    volume: str
    path: Path  # actual path to the file


def get_image_from_subfolders(base_path: Path, extension: str = "png"):
    volume_mappings: dict[str, list[ImageData]] = {}
    for volume in base_path.iterdir():
        if not volume.is_dir():
            continue
        stem_name: str = volume.stem
        if not stem_name.startswith("v"):
            continue
        if stem_name.endswith("_pgc"):
            continue
        if stem_name not in volume_mappings:
            volume_mappings[stem_name] = []
        for image in volume.glob(f"*.{extension}"):
            volume_mappings[stem_name].append(ImageData(image.stem, stem_name, image))
    return volume_mappings


def _rejoin_planes(luma: vs.VideoNode, chroma: list[vs.VideoNode]):
    if len(chroma) == 0:
        return luma
    return join([luma, *chroma])


def denoise_step(
    clip: vs.VideoNode,
    tr: int = 2,
    thSAD: int = 180,
    sigma: float = 0.88,
    two_step: bool = True,
    split_planes: bool = True,
):
    luma_planar: vs.VideoNode = None
    chroma_planar: list[vs.VideoNode] = []
    if split_planes:
        split_planes = split(clip)
        if len(split_planes) == 1:
            # Gray world image
            luma_planar = clip
        else:
            luma_planar = split_planes[0]
            chroma_planar = split_planes[1:]
    else:
        luma_planar = clip

    filt_dn_smd = SMDegrain(luma_planar, tr=tr, thSAD=thSAD, thSADC=0, RefineMotion=True, prefilter=2)
    if not two_step:
        if split_planes:
            return _rejoin_planes(filt_dn_smd, chroma_planar)
        return filt_dn_smd
    filt_dn_bm3d = core.bm3dcuda.BM3D(
        depth(luma_planar, 32), ref=depth(filt_dn_smd, 32), sigma=sigma, radius=2
    ).bm3d.VAggregate(radius=2)
    if split_planes:
        return _rejoin_planes(filt_dn_bm3d, chroma_planar)
    return filt_dn_bm3d


def prepare_image(path: Path):
    clip = core.imwri.Read(str(path))
    or_w, or_h = clip.width, clip.height
    if or_h % 2 != 0:
        clip = core.resize.Spline36(clip, clip.width + 1, clip.height + 1)
    clip = ToYUV(clip)
    clip = depth(clip, 16)
    if or_h % 2 != 0:
        clip = core.resize.Spline36(clip, or_w, or_h)
    return clip


def postprocess_image(clip: vs.VideoNode):
    or_w, or_h = clip.width, clip.height
    if or_h % 2 != 0:
        clip = core.resize.Spline36(clip, clip.width + 1, clip.height + 1)
    clip = ToRGB(clip)
    clip = depth(clip, 8)
    if or_h % 2 != 0:
        clip = core.resize.Spline36(clip, or_w, or_h)
    return clip


def _find_image_clip(clip: vs.VideoNode, clip_sets: list[vs.VideoNode]):
    # Find the clip that might fits in the clip_sets
    cw, ch = clip.width, clip.height
    for i, c in enumerate(clip_sets):
        if c.width == cw and c.height == ch:
            return i
    # No match, so we need to create a new clip
    return None


def bulk_open_images(images: list[ImageData]) -> tuple[list[vs.VideoNode], list[list[ImageData]]]:
    # We might have an images at different sizes, so we want to make a new clip format
    xclip_sets: list[vs.VideoNode] = []
    image_pairs: list[list[ImageData]] = []
    for image in images:
        clip = prepare_image(image.path)
        if (idx := _find_image_clip(clip, xclip_sets)) is not None:
            xclip_sets[idx] = xclip_sets[idx] + clip
            image_pairs[idx].append(image)
        else:
            xclip_sets.append(clip)
            image_pairs.append([image])
    return xclip_sets, image_pairs
