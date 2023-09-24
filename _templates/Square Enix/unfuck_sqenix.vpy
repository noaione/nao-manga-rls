import sys
from pathlib import Path

import vapoursynth as vs
from vapoursynth import core
import mvsfunc as mvf
import havsfunc as hvf
import n4ofunc as nao
import fvsfunc as fvf


def prepare(clip: vs.VideoNode):
    if clip.height % 2 != 0:
        clip = core.resize.Spline36(clip, clip.width + 1, clip.height + 1)
    as_yuv = mvf.ToYUV(clip)
    as_yuv = mvf.Depth(as_yuv, 16)
    # Resize if needed

    return as_yuv


def finalize(clip: vs.VideoNode, original_w: int, original_h: int):
    final_clip = mvf.Depth(clip, 8)
    final_clip = mvf.ToRGB(final_clip)
    if clip.height != original_h:
        final_clip = core.resize.Spline36(final_clip, original_w, original_h)
    return final_clip


def filterchain(source: vs.VideoNode, compare: bool = False):
    width = source.width
    height = source.height
    if source.format.color_family != vs.YUV:
        source = prepare(source)

    dehalo = hvf.FineDehalo(source, rx=2, thlimi=80, thlima=120, contra=0.5)
    aa = hvf.daa(dehalo, nsize=4, nns=2, qual=2, opencl=False, device=0)
    dehalo2 = hvf.FineDehalo(aa)
    #linedark = hvf.FastLineDarkenMOD(dehalo2)
    #deblur = core.cas.CAS(linedark, 0.7, [0, 1, 2])
    if compare:
        return nao.compare(
            [
                source,
                dehalo,
                aa,
                dehalo2,
                #linedark,
                #deblur,
            ],
            interleave_only=True,
            identity=True
        )
    return finalize(dehalo2, width, height)


folder_to_process = Path(sys.argv[1])
cwd = Path.cwd()
# folder_to_process = Path("_temp2")
images_to_process = list(filter(lambda x: x.suffix.strip(".").lower() in ["png", "jpg", "jpeg"], list(folder_to_process.glob("*"))))

base_image = None
actual_filename: list[str] = []
print(f"Reading {len(images_to_process)} images...")
for image in images_to_process:
    read_img = core.imwri.Read(str(image))
    filt_img = filterchain(read_img)
    actual_filename.append(image.stem)
    if base_image is None:
        base_image = filt_img
    else:
        base_image = base_image + filt_img


print("Writing images...")
time_to_write = core.imwri.Write(base_image, "PNG", "_page%03d.png", 0, 100)
time_to_write.set_output(0)

if __name__ == "__main__":
    for i in range(time_to_write.num_frames):
        print(f"Writing frame {i}")
        time_to_write.get_frame(i)
        denoise_fn = cwd / f"{actual_filename[i]}.png"
        source_fn = cwd / f"_page{i:03d}.png"
        print(f"  Renaming {source_fn} to {denoise_fn}")
        source_fn.rename(denoise_fn)

#cover = core.imwri.Read("cover_i-btgotg_06-1x_Saiyajin_DeJPEG_300000_G.png")
#page001 = core.imwri.Read("i-btgotg_06_001-1x_Saiyajin_DeJPEG_300000_G.png")
#page002 = core.imwri.Read("i-btgotg_06_002-1x_Saiyajin_DeJPEG_300000_G.png")

#merged = filterchain(cover) + filterchain(page001) + filterchain(page002)
#merged_out = core.imwri.Write(merged, "PNG", "_page%03d.png", 0, 100)

#merged_out.set_output(0)