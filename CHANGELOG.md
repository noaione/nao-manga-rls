# Changelog

## Unreleased

**New Features**
- `nmanga ort infer` - `--tta` averages every batch over its 8 flips/rotations (test-time augmentation, about 8x slower). The orchestrator `denoise`/`upscale` actions expose it as `tta`

**Changes**
- `torch` backend now cache the compiled model under `$DATA_DIR/torch_inductor` (`$DATA_DIR` is the same as `ort` backend)
- `ort infer` and the orchestrator `denoise`/`upscale` actions tile with the waifu2x overlap-add: `tile_overlap` is now the real output overlap (stride `tile_size - tile_overlap`), so the same value runs fewer, cheaper tiles and seams blend with complementary linear ramps
- Overlapping ONNX runs replicate the page edge instead of padding it with `--background`, so the outer tiles no longer pick up a black/white border (the flag still applies to non-overlapping runs)

**Fixes**
- ONNX inference feeds the precision the model declares, so an fp16 and an fp32 file are each fed as they expect instead of following the caller's guess
- ONNX inference pads a short final batch up to a static model's declared batch, so static TensorRT/TRT-RTX engines no longer reject the last tiles
- Contrast stretching no longer divides by zero on a single-colour page

## [0.31.0] (2026-11-08)

**New Features**
- `nmanga ort infer` - One command replaces `nmanga upscale-trt` and `nmanga denoise-trt`, which only ever ran the same tiled inference. The scale factor decides the wording: x1 denoises, anything bigger upscales
  - `-be/--backend` names the execution provider in nmanga terms (`auto`, `trt-rtx`, `trt`, `cuda`, `amdgpu`, `migraphx`, `openvino`, `coreml`, `mlx`, `webgpu`). Naming one makes it mandatory, and `--strict-ep` now only applies to `auto`
  - `nmanga ort backends` prints what the installed runtime can use, with `--raw` adding the provider and device names. `-rtx/--with-nvrtx` becomes `--backend trt-rtx`
- `nmanga torch infer` - Run a checkpoint with PyTorch instead of ONNX Runtime
  - `--preset fast` applies the lean FDAT rewrites, `--preset original` runs the model untouched. FP32, batch 1, no autocast
  - Uses the validated torch tile grid, where the overlap is the total between two tiles, not the ONNX tiler
  - `--device auto|cpu|cuda:N`, optional `torch.compile` on CUDA, folder or `--recursive` input, PNG output
- `nmanga torch to-onnx` - Moved under the `torch` group, same options
- `nmanga lookup imagesize` - Lookup image sizes in an archive or folder
- The ONNX session builders, engine cache, tiling, ORT initialization and Windows ML discovery moved into `nmanga.ml`, with `nmanga.denoiser`, `nmanga.winml` and `nmanga.lazy` left as thin forwards

**Changes**
- `nmanga denoise`, `nmanga denoise-trt`, `nmanga upscale-trt` and `nmanga identify-quality` are gone, along with the `w2x_trt_path` setting. `prepare_model_runtime` is gone too, so the TensorRT builder is the only session path
- Orchestrator `denoise`/`upscale` actions - `precompiled` and `rtx_ep` become `backend`, and `data_type` defaults to `fp16`
- Orchestrator ML actions - Stop requiring `tensorrt`, `torch` and `onnx` as tools
- `safetensors` is an explicit `denoiser` dependency, `numpy` and `scipy` are regular ones, and the `autostuff` extra is gone
- `denoiser` installs `triton-windows` on Windows, which `torch.compile` needs there
- Split the machine learning extras by vendor. `denoiser` keeps the vendor neutral stack, and `denoiser-ort`, `denoiser-nvidia`, `denoiser-intel`, `denoiser-wgpu` or `denoiser-mlx` each add a runtime
  - `denoiser-nvidia` replaces the stock `onnxruntime`, so it conflicts with `denoiser-ort` and `denoiser-mlx`
  - No `denoiser-amd` yet, the `amdgpu` backend name is reserved

**Fixes**
- The GPU memory probe no longer raises when torch has no usable CUDA device, so the plugin extras fall back to the default workspace

## [0.30.0] (2026-11-07)

**New Features**
- `nmanga orchestra run-target` - Run the whole action chain on one target directory without listing it in `volumes`
  - The target has to live inside `base_path`, and actions that need volume metadata (`rename`, `spreads`) are refused up front
  - `--dry-run` prints what every action would do without touching a file
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Add `-ws/--workspace-size`, `-cg/--cuda-graph` and `-sep/--strict-ep` so a comparison can reproduce the build settings
  - `nmanga denoise-trt` also grows `-rtx/--with-nvrtx` and `-qs/--quant-size`, which route it through the TensorRT RTX builder
- Orchestrator `denoise`/`upscale` actions - Add `workspace_size`, `enable_cuda_graph` and `strict_ep` fields

**Fixes**
- `nmanga upscale-trt` and `nmanga denoise-trt` - Build the TensorRT profiles from the dimensions the model declares, so a fixed `512x512` input no longer gets a contradictory default profile
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Key the engine cache on the graph contents, precision, workspace, CUDA graph setting and TF32 policy instead of the pathname
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Probe the scale factor with the batch size the engine was built for
- ONNX Runtime telemetry is disabled in the shared loader

## [0.29.0] (2026-10-29)

**New Features**
- Orchestrator `autolevel` and `posterize` actions - The `v3` and `vapoursynth` modes are now implemented, backed by the `nimages` plugin like `nmanga autolevel3` and `nmanga posterize2`, and carry the `cache_mb` and `prefetch` options
- `nmanga posterize2` - Metric guided posterization. `--bits` takes one depth or several, and several turn the command into a fan-out over the `vship` metrics
  - `--bits 2-5` or `--bits 2,3,4` posterize every candidate depth, score them per metric, and keep the least degraded depth every thresholded metric accepts
  - `--metrics` picks what scores the candidates (`ssimulacra2`, `butteraugli`, `cvvdp`), `--metric-min` sets a threshold per metric, and `--bits-policy` chooses between "fewest bits that passes" and the order given
  - A page no candidate passes is copied from the source, and every candidate hangs off one shared decode, so a volume costs one decode rather than one per page
  - One depth with no `--metrics` scores nothing, so `--bits 4` stays the fixed depth posterize it always was
  - **Breaking:** `--use-ssimulacra2` and `--ssim-min` are gone, their equivalent is `--bits <depth> --metrics ssimulacra2`
- `nmanga autolevel3`, `nmanga posterize2`, `nmanga deblur2` - The PNG encode and the publish now belong to `imgseqs.PNGWrite`, so the commands write the destination directly with no staging directory, and `--threads` is the number of open write requests
  - `nmanga deblur2` no longer fails on a 16-bit source, and the writer publishes truthful PNG metadata (`sBIT`, `cICP`)
- `nmanga autolevel3` - One output mode replaces `--keep-colorspace` and `--force-gray`
  - `--color/--gray`, colour by default: a page keeps its own container's colour, and `--gray` writes one plane per page
  - The analyzer reads the source pages directly, `--prefetch` (default 16) sizes the decoder lookahead, `--cache` bounds the frame cache, and `-v` reports timings, levels and peak flags
- `nmanga autolevel3` and `nmanga posterize2` - `--gpu` runs the analysis and leveling on the device. It needs `vapoursynth-nimages` 0.2.1 and a Vulkan 1.4 device, and stays off by default
- `nmanga posterize2` - Output comes from the chosen `Posterize` output instead of a float round trip, and a volume that resolves two pages to one destination is refused before the first write

**Changes**
- `nmanga autolevel3` - **Breaking:** `--keep-colorspace`, `--force-gray` and `-f/--format` are gone. Output is PNG only and the default flips from gray to colour

**Fixes**
- `nmanga shiftname` and the orchestrator `shift_rename` action - Work out a safe rename order automatically, and fail before touching anything when no safe order exists
  - A right shift is renamed descending and a left shift ascending, and `-r/--reverse` keeps meaning "number the pages bottom up"
  - A rename that fails partway is rolled back, an existing file is never replaced, and `-n/--dry-run` prints the order
- `nmanga shiftname` - Report a non-zero exit code when a rename cannot be performed
- `nmanga-gui` - Check the chapter ranges against the source images, forget the previous source when a new one fails to load, and spell out skipped pages instead of collapsing them into one label

## [0.28.0] (2026-09-27)

**New Features**
- `nmanga autolevel3` and `nmanga posterize2` - New VapourSynth based autolevel and posterize commands
- `nmanga-gui` - Optional PySide6 GUI for the manual split feature, install with the `gui` extra
  - Load a folder of images or an archive and see the pages as a thumbnail grid
  - Drag a folder or archive from the file manager and drop it anywhere on the window to load it
  - Drag-select pages into chapter ranges, type the ranges directly, or press Add to append the next range
  - Skip pages with a comma separated list such as `1,5-20`, which is kept visible in the range and in the
    uncovered page warnings
  - Preview which chapter owns each page, and get warned about uncovered, overlapping, or missing pages,
    with the page numbers listed
  - Carry a volume per chapter, so an omnibus can be split into `v01-02` chapters
- `nmanga posterize` and `nmanga autolevel2` - Add faster color detection and a VapourSynth decode path

**Fixes**
- `nmanga upscale-trt` and `nmanga denoise-trt` - Clamp the batch size down to the one hardcoded in the model
  (e.g. an input shape of `[1, 3, height, width]`) instead of failing with an `INVALID_ARGUMENT` error
- `nmanga upscale-trt` and `nmanga denoise-trt` - Report a proper error when the model declares a fixed tile size
  that differs from the requested one

## [0.27.1] (2026-09-12)

**Changes**
- `nmanga spreads join` - Add `--gap`
- `--recursive` now skips folders that hold no images, and the posterize commands gained the option
- `nmanga epub` - Add EPUB diffing

**Fixes**
- Orchestrator ML actions - Fix `rtx_ep` not being applied
- `spreads split` - Fix the split output and the model cache directory

## [0.27.0] (2026-07-28)

**New Features**
- `nmanga jxlify` - Convert images to JPEG XL with `cjxl`, plus a `cjxl_path` configuration entry

## [0.26.0] (2026-07-22)

**New Features**
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Add `-to/--tile-overlap`
- Orchestrator - Omnibus volume numbering
- `nmanga epub render` - Render EPUB pages to images with Playwright, install with the `epub-render` extra
- Faster color detection, and `--fast` for the color lookup

**Fixes**
- Fix the TRT-RTX GPU capability check to query the requested device instead of the current one

## [0.25.0] (2026-06-28)

**New Features**
- `nmanga imops` - Image operations command
- `nmanga rescale` - Add `-f/--force`
- Orchestrator `posterize` action - SSIMULACRA2 check

**Changes**
- The default thread count is now half of the CPU threads

## [0.24.0] (2026-06-07)

**New Features**
- `nmanga install-completion` - Install shell completion for the current shell
- New ML actions for the orchestrator
- MiGraphX and OpenVINO can be enabled through WinML
- `nmanga lookup` - Color page moving
- Improved autolevel

**Changes**
- Python 3.12 is the new minimum

## [0.23.0] (2026-03-11)

**New Features**
- `nmanga orchestra` - Create/run a JSON file that will run multiple functions in nmanga together
- `nmanga upscale-trt` - Upscale images with an ONNX model
- `nmanga rescale` - Rescale images with a chosen kernel
- `nmanga releases` and `nmanga releasesch` - Auto quality detection, backed by the configured LQ/HQ thresholds
- `nmanga epub-extract` - Extract images from an EPUB
- Add tools to check for broken images
- Add a quick grayscale image lookup/mover
- Reworked CLI display with `rich-click` for better experience
- Implemented threaded tagging for better performance
- Use `rich` progress bar for better progress display
- Implement better support for outputing to STDOUT/STDERR in threaded functions
- Add `--recursive` option to the following commands:
  - `nmanga denoise-trt`
  - `nmanga autolevel2`
  - `nmanga jpegify`
- `nmanga manualsplit` now accepts a folder of images, not just an archive
- `nmanga manualsplit` supports a volume per chapter range (including omnibus ranges such as `1-2`)
  - Use `-vol 1-2` for a whole-omnibus default, or `--per-chapter-volume` to set one per chapter
- `nmanga manualsplit` accepts a comma separated list of pages (for example `1,5-20`) to skip pages inside a
  chapter range; a comma turns off the bare number "until the end" form so the last page must be given

**Changes**
- Improve manual split filename matcher
- Move denoise and spreads joiner to the main nmanga module instead of in CLI part only
- Adjust image gray level peak detection
- Allow modifying compression level when packing archives
  - Use `-cl/--compression-level` option in `nmanga pack` and `nmanga packepub`
  - Alternatively, use `compress_level` in the `pack` orchestrator action
- Move from standard `click` to `rich-click` for better CLI experience
- Make threaded worker run function serially when threads is set to 1 to avoid overhead

## [0.22.0] (2025-10-20)

**Changes**
- Moved to the `uv` ecosystem, with a `uv.lock` and CI installing through `uv`
- More typing fixes, and ruff runs over the whole package

**Fixes**
- Fix some issues with PDF images extraction and compositing

## [0.21.0] (2025-10-13)
**New Features**
- Add `posterize` and `autoposterize` command to reduce the number of shades in the image
  - Posterize: Reduce to fixed number of shades (2, 4, 8, 16, 32, 64, 128)
  - Autoposterize: Reduce to optimal number of shades based on threshold percentage (default 0.5%)
- Add `--use-pil` to `nmanga spreads join` command to use Pillow instead of ImageMagick for joining spreads

**Changes**
- Make `magick convert` to be `magick` instead since it's deprecated in ImageMagick 7+
- Properly say how many images is copied without autolevel
- Force use UTF-8 when opening archive metadata
- Add `--keep-colorspace` on `autolevel2` command to keep original colorspace
- Add more detailed info on `denoise-trt` command
- Fix issue with `nmanga merge` command not working properly

**Refactor**
- Move to Python 3.10+ as minimum and change typing to use built-in generics

## [0.20.0] (2025-08-29)
**New Features**
- PDF handling via `nmanga pdf`
  - Added `pymupdf` (fitz) dependency for PDF handling
  - Add `nmanga pdf extract` command to extract images from PDFs
  - Add `nmanga pdf identify` command to show PDF DPI and page size information
  - Add `nmanga pdf export` command to export PDF pages as images
- (Experimental) automatic color leveling with Pillow via `nmanga autolevel2`
  - There might be miniscule difference with ImageMagick version, so YMMV

**Fixes**
- Make manga title optional in `nmanga shiftname` command
- Add missing proper command handler in `nmanga denoise-trt` command

**Refactor**
- Remove `nmanga detect-grayscale` command as it's not really useful
- Raise the minimum Pillow image size
- Bump all dependencies

## [0.19.0] (2025-08-27)
**New Features**
- Add `nmanga denoise-trt` command to denoise images via TensorRT/ONNX Runtime (Experimental)
  - Needs `onnxruntime`, `onnxruntime-gpu`, `tensorrt` and all the NVIDIA related dependencies installed

**Fixes**
- Fix missing `p` prefix on page numbering when using `nmanga shiftname`
- Fix broken recommendation

**Refactor**
- Use proper single thread when threads is set to 1

## [0.18.0] (2025-08-27)
**New Features**
- Add `nmanga denoise` command to denoise images via ImageMagick
  - Also include `nmanga identify-quality` to identify image quality via ImageMagick
- Add `nmanga detect-grayscale` to detect grayscale images via HSV color space calculation
  - Needs `scikit-image`, `numpy`, and `Pillow` installed
- Add `nmanga shiftname` to bulk rename files with padded number
  - Useful for renaming files like `img_1.jpg`, `img_2.jpg`, ... to `p000.jpg`, `p001.jpg`, ...
  - Supports adding manga title and volume to the name.

**Fixes**
- Fix `--format` option in `nmanga autolevel` command

## [0.17.0] (2025-08-27)
**New Features**
- Add `nmanga autolevel` command to auto-level images via ImageMagick

## [0.16.5] (2025-08-22)
**Fixes**
- Support 4-digits for page number, 4 digits for chapter, and 3 digits for volume
- Fix weird volume number being used

## [0.16.4] (2025-05-27)
**Refactor**
- Better handling of oneshot in `nmanga releases` command

**Build**
- Bump all dependencies

## [0.16.3] (2024-06-20)
**Fixes**
- Manual split on page number mode not working as intended.

## [0.16.2] (2024-06-10)
**Fixes**
- Use `.webp` extension on spreads join properly

## [0.16.1] (2024-05-30)
**Fixes**
- Regex escape publication type like `raw-d` and more

## [0.16.0] (2024-04-18)
**New Features**
- **[BREAKING CHANGES]** Remove `--png-tag` option, use `exiftool` instead to tag PNG images

**Changes**
- Apply `--extra-meta` or `extra_archive_metadata` to manga title
  - Before: `Test Title - cXXX (vXX) - pXXX [CH Extra] [dig] [Publisher] [Ripper]`
  - After: `Test Title [VOL Meta] - cXXX (vXX) - pXXX [CH Meta] [dig] [Publisher] [Ripper]`

## [0.15.0] (2024-04-11)
**New Features**
- Add new option `-ex/--extra-meta` to add extra text before publication type on the archive filename
  - Use case: `Test Manga v01 (20xx) (Omnibus 2-in-1) (Digital) (nao)`
  - Command used: `nmanga pack -t "Test Manga" -ex "Omnibus 2-in-1" -vol 1 -br round -c nao`
- Added `digital-raw` publication type which will use `raw-d` in image filename and `Digital` in archive filename

**Changes**
- [BREAKING CHANGES] Make `format_archive_filename` and `format_daiz_like_filename` parameters to be all positional.

## [0.14.1] (2024-03-21)
**Changes**
- Fix `filename` attribute missing
- Refactor a bit more on `pathlib.Path` usages

## [0.14.0] (2024-03-21)
**Changes**
- Bump all dependencies
  - Also fix problem with unrar2 failed to install
- Lint code with ruff

## [0.13.0] (2024-01-11)
**New Features**
- Modify file timestamp with `nmanga timewizard`

## [0.12.0] (2023-10-30)
**New Support/BREAKING CHANGES**
- Use `unrar2-cffi` for modern Python
- Drop Python <3.9

## [0.11.3] (2023-09-18)
**Fixes**
- Do not remove transparency on PNG

## [0.11.2] (2023-08-09)
**Fixes**
- exiftool not detected properly
- Fix wrong exception catch for catching command timeout
- Do an image count check when checking for all image filename validity on `releases` and `releasesch`

## [0.11.1] (2023-08-08)
**Fixes**
- Re-add missing parameters for old alpha version of `pingo`

## [0.11.0] (2023-08-08)
**New Features**
- Support pingo 1.x
  - Default main command are: `pingo -notrans -notime -lossless -s4`
    - With aggresive mode:
      - JPEG: Remove `-lossless` and add `-q=97`
      - WEBP: Remove `-lossless` and add `-webp`
- Enable PNG tagging via command option

## [0.10.3] (2023-06-29)
**Fixes**
- Properly support special volume scanning/collection

## [0.10.2] (2023-06-29)
**New Features**
- Inject metadata to PNG via bytes injection
  - To enable, use `nmanga config` and configure the experimental part
- Raw tagging with provided metadata instead of custom metadata formatting we use
- Add support for special numbered volume (v01.5) etc.

## [0.10.1] (2023-06-15)
**Fixes**
- Tagging for chapter release is incorrect

## [0.10.0] (2023-06-14)
**Refactor**
- Move `nmanga.cli.constants` to `nmanga.constants` and move the original into `nmanga._metadata`

**New Features**
- Allow the "unsecure" characters back on Linux

## [0.9.1] (2023-05-26)
**Refactor**
- Restructure the module structure
- Add initial tests file

**Fixes**
- More filename format fix
- Wrong typing

## [0.9.0] (2023-05-17)
**BREAKING CHANGES**
- Deprecated `--is-high-quality` in favor of `--quality` for marking LQ/HQ image

**Fixes**
- Publication type not being written
- Extra mapping is now gated behind empty check before processing
- Fix typo when it should use `manga_volume_text` instead of `manga_volume

## [0.8.2] (2023-05-17)
**New Features**
- Add configurable option for default publication type

**Refactor**
- Create a function for creating filename format, simplify it too.
- This changes the formatting around a bit.

## [0.8.1] (2023-05-17)
**New Features**
- Add `nmanga releasesch` to rename a single chapter.
  - Same with spreads, it needs a minimum of `p000` for filename.

## [0.8.0] (2023-05-15)
**New Features**
- Add support for archiving to `epub` (ZIP archive) or `cb7` (7z archive)
- Add support for other publication type in the filename
  - magazine
  - web
  - scan/c2c

**Fixes**
- Move all `pack` command to it's own module (`nmanga.cli.archive`)

## [0.7.2] (2023-05-09)
**New Features**
- Add `chapter_add_c_prefix` default behaviour configuration (add `c` prefix to chapter packing `c001` instead of just `001`)
- Add `chapter_special_tag` default behaviour configuration (use `#` instead of `x` as special chapter number separator)

**Fixes**
- Fix typo in config warning decorator

## [0.7.1] (2023-05-09)
**New Features**
- Add `nmanga packcomment` to modify archive comment

## [0.7.0] (2023-05-01)
**New Features**
- Add `nmanga configure` to configure some defaults behaviour and executable path
- Allow `--revision` parameter in image tagging

## [0.6.0] (2023-04-07)
**BREAKING CHANGES**
- Move `spreads` command to `spreads join` to join spreads

**New Features**
- Add new feature called `spreads split` to split joined spreads

## [0.5.0] (2023-03-25)
**BREAKING CHANGES**
- Remove `nmanga level` command, just use `magick mogrify` instead.

## [0.4.4] (2023-02-21)
**Fixes**
- Support for web/mag/c2c/paper/scan in `autosplit` checking

## [0.4.3] (2023-02-21)
**New Features**
- Better optimizer command output
- Allow for oneshot volume in `releases` command
- Allow for adding revision number to `pack`/`releases`

**Fixes**
- Simplify or unify some options for less repeat.

## [0.4.2] (2023-01-27)
**Fixes**
- Fix broken spreads join command

## [0.4.1] (2023-01-27)
**Fixes**
- Do not use recursive glob when packing
- Support for text behind `p001` that should be added back

## [0.4.0] (2023-01-18)
**New Features**
- Add `nmanga packepub` command for novel packing
- Allow a really basic page numbering for `spreads join` (Basically needs minimum of `p000` or `p001-002`)

**Fixes**
- Print the final text when stopping console status

## [0.3.4] (2023-01-18)
**Fixes**
- Expect only float or int for `--volume` and `--chapter`

## [0.3.3] (2023-01-17)
**New Features**
- Add `--chapter` option for `nmanga pack` that allow for single chapter packing naming.

## [0.3.2] (2023-01-17)
**Fixes**
- Extend the image mimetype checking for modern image formats.

## [0.3.1] (2022-12-27)
**Fixes?**
- Only allow `.jpg` files for optimizing/tagging

## [0.3.0] (2022-12-26)
**New Features**
- Add support for optimizing images via `pingo`
- Include the command `nmanga optimize` too.

## [0.2.4] (2022-12-26)

**Fixes**
- Properly handle single chapter float number

## [0.2.3] (2022-12-25)

**New Features**
- Add floating chapter (extra/omake/bonus) number into the image filename if needed. (Mainly for numbering below `.5`)

**Fixes**
- Force `--volume` as required in `nmanga tag`

## [0.2.2] (2022-12-21)

**New Features**
- Add separate `nmanga tag` command.

**Fixes**
- Fix utf-8 handling for zipfile that does not use it properly via [`ftfy`](https://pypi.org/project/ftfy/)

## [0.2.1] (2022-12-07)

**Build**
- Bump requirements

## [0.2.0] (2022-08-30)

**New Features**
- All-in-One class for archive opener and new command ([#2](https://github.com/noaione/nao-manga-rls/pull/2))

## [0.1.1] (2022-08-03)

**Fixes**
- Fix `.rar` file unable to be opened.

## [0.1.0] (2022-07-06)

Rewritten every thing as a module with `nmanga` namespace.
