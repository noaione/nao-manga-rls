# Changelog

## 0.1.0

Rewritten every thing as a module with `nmanga` namespace.

### 0.1.1

**Fixes**
- Fix `.rar` file unable to be opened.

## 0.2.0

**New Features**
- All-in-One class for archive opener and new command ([#2](https://github.com/noaione/nao-manga-rls/pull/2))

### 0.2.1

**Build**
- Bump requirements

### 0.2.2

**New Features**
- Add separate `nmanga tag` command.

**Fixes**
- Fix utf-8 handling for zipfile that does not use it properly via [`ftfy`](https://pypi.org/project/ftfy/)

### 0.2.3

**New Features**
- Add floating chapter (extra/omake/bonus) number into the image filename if needed. (Mainly for numbering below `.5`)

**Fixes**
- Force `--volume` as required in `nmanga tag`

### 0.2.4

**Fixes**
- Properly handle single chapter float number

## 0.3.0
**New Features**
- Add support for optimizing images via `pingo`
- Include the command `nmanga optimize` too.

### 0.3.1
**Fixes?**
- Only allow `.jpg` files for optimizing/tagging

### 0.3.2
**Fixes**
- Extend the image mimetype checking for modern image formats.

### 0.3.3
**New Features**
- Add `--chapter` option for `nmanga pack` that allow for single chapter packing naming.

### 0.3.4
**Fixes**
- Expect only float or int for `--volume` and `--chapter`

## 0.4.0
**New Features**
- Add `nmanga packepub` command for novel packing
- Allow a really basic page numbering for `spreads join` (Basically needs minimum of `p000` or `p001-002`)

**Fixes**
- Print the final text when stopping console status

### 0.4.1
**Fixes**
- Do not use recursive glob when packing
- Support for text behind `p001` that should be added back

### 0.4.2
**Fixes**
- Fix broken spreads join command

### 0.4.3
**New Features**
- Better optimizer command output
- Allow for oneshot volume in `releases` command
- Allow for adding revision number to `pack`/`releases`

**Fixes**
- Simplify or unify some options for less repeat.

### 0.4.4
**Fixes**
- Support for web/mag/c2c/paper/scan in `autosplit` checking

## 0.5.0
**BREAKING CHANGES**
- Remove `nmanga level` command, just use `magick mogrify` instead.

## 0.6.0
**BREAKING CHANGES**
- Move `spreads` command to `spreads join` to join spreads

**New Features**
- Add new feature called `spreads split` to split joined spreads

## 0.7.0
**New Features**
- Add `nmanga configure` to configure some defaults behaviour and executable path
- Allow `--revision` parameter in image tagging

### 0.7.1
**New Features**
- Add `nmanga packcomment` to modify archive comment

### 0.7.2
**New Features**
- Add `chapter_add_c_prefix` default behaviour configuration (add `c` prefix to chapter packing `c001` instead of just `001`)
- Add `chapter_special_tag` default behaviour configuration (use `#` instead of `x` as special chapter number separator)

**Fixes**
- Fix typo in config warning decorator

## 0.8.0
**New Features**
- Add support for archiving to `epub` (ZIP archive) or `cb7` (7z archive)
- Add support for other publication type in the filename
  - magazine
  - web
  - scan/c2c

**Fixes**
- Move all `pack` command to it's own module (`nmanga.cli.archive`)

### 0.8.1
**New Features**
- Add `nmanga releasesch` to rename a single chapter.
  - Same with spreads, it needs a minimum of `p000` for filename.

### 0.8.2
**New Features**
- Add configurable option for default publication type

**Refactor**
- Create a function for creating filename format, simplify it too.
- This changes the formatting around a bit.

## 0.9.0
**BREAKING CHANGES**
- Deprecated `--is-high-quality` in favor of `--quality` for marking LQ/HQ image

**Fixes**
- Publication type not being written
- Extra mapping is now gated behind empty check before processing
- Fix typo when it should use `manga_volume_text` instead of `manga_volume

### 0.9.1
**Refactor**
- Restructure the module structure
- Add initial tests file

**Fixes**
- More filename format fix
- Wrong typing

## 0.10.0
**Refactor**
- Move `nmanga.cli.constants` to `nmanga.constants` and move the original into `nmanga._metadata`

**New Features**
- Allow the "unsecure" characters back on Linux

### 0.10.1
**Fixes**
- Tagging for chapter release is incorrect


### 0.10.2
**New Features**
- Inject metadata to PNG via bytes injection
  - To enable, use `nmanga config` and configure the experimental part
- Raw tagging with provided metadata instead of custom metadata formatting we use
- Add support for special numbered volume (v01.5) etc.

### 0.10.3
**Fixes**
- Properly support special volume scanning/collection

## 0.11.0
**New Features**
- Support pingo 1.x
  - Default main command are: `pingo -notrans -notime -lossless -s4`
    - With aggresive mode:
      - JPEG: Remove `-lossless` and add `-q=97`
      - WEBP: Remove `-lossless` and add `-webp`
- Enable PNG tagging via command option

### 0.11.1
**Fixes**
- Re-add missing parameters for old alpha version of `pingo`

### 0.11.2
**Fixes**
- exiftool not detected properly
- Fix wrong exception catch for catching command timeout
- Do an image count check when checking for all image filename validity on `releases` and `releasesch`

### 0.11.3
**Fixes**
- Do not remove transparency on PNG

## 0.12.0
**New Support/BREAKING CHANGES**
- Use `unrar2-cffi` for modern Python
- Drop Python <3.9

## 0.13.0
**New Features**
- Modify file timestamp with `nmanga timewizard`

## 0.14.0
**Changes**
- Bump all dependencies
  - Also fix problem with unrar2 failed to install
- Lint code with ruff

### 0.14.1
**Changes**
- Fix `filename` attribute missing
- Refactor a bit more on `pathlib.Path` usages

## 0.15.0
**New Features**
- Add new option `-ex/--extra-meta` to add extra text before publication type on the archive filename
  - Use case: `Test Manga v01 (20xx) (Omnibus 2-in-1) (Digital) (nao)`
  - Command used: `nmanga pack -t "Test Manga" -ex "Omnibus 2-in-1" -vol 1 -br round -c nao`
- Added `digital-raw` publication type which will use `raw-d` in image filename and `Digital` in archive filename

**Changes**
- [BREAKING CHANGES] Make `format_archive_filename` and `format_daiz_like_filename` parameters to be all positional.

## 0.16.0
**New Features**
- **[BREAKING CHANGES]** Remove `--png-tag` option, use `exiftool` instead to tag PNG images

**Changes**
- Apply `--extra-meta` or `extra_archive_metadata` to manga title
  - Before: `Test Title - cXXX (vXX) - pXXX [CH Extra] [dig] [Publisher] [Ripper]`
  - After: `Test Title [VOL Meta] - cXXX (vXX) - pXXX [CH Meta] [dig] [Publisher] [Ripper]`

### 0.16.1
**Fixes**
- Regex escape publication type like `raw-d` and more

### 0.16.2
**Fixes**
- Use `.webp` extension on spreads join properly

### 0.16.3
**Fixes**
- Manual split on page number mode not working as intended.

### 0.16.4
**Refactor**
- Better handling of oneshot in `nmanga releases` command

**Build**
- Bump all dependencies

### 0.16.5
**Fixes**
- Support 4-digits for page number, 4 digits for chapter, and 3 digits for volume
- Fix weird volume number being used

## 0.17.0
**New Features**
- Add `nmanga autolevel` command to auto-level images via ImageMagick

## 0.18.0
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

## 0.19.0
**New Features**
- Add `nmanga denoise-trt` command to denoise images via TensorRT/ONNX Runtime (Experimental)
  - Needs `onnxruntime`, `onnxruntime-gpu`, `tensorrt` and all the NVIDIA related dependencies installed

**Fixes**
- Fix missing `p` prefix on page numbering when using `nmanga shiftname`
- Fix broken recommendation

**Refactor**
- Use proper single thread when threads is set to 1

## 0.20.0
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

## 0.21.0
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

## Unversioned
**New Features**
- `nmanga torch infer` - Run a checkpoint model directly with PyTorch. `resselt` detects the architecture and
  loads the weights, so no ONNX Runtime session or TensorRT engine is involved
  - `--preset fast` applies the adapted lean FDAT rewrites (in-place GELU, lean spatial attention, lean FFN and
    lean DAT block) to an independent copy of the model, `--preset original` runs the untouched model. FP32 only,
    batch 1, no autocast; the stored checkpoint dtypes are read from the tensors, not from the filename
  - The tiling is the validated JaNai one, not the ONNX one: the overlap is the total overlap between two tiles and
    partial edge tiles are replicate-padded before the prediction, so `--tile-size 256 --tile-overlap 32` is a
    224px stride and does not produce the same page as the ONNX tiler's 192px stride
  - `--device auto|cpu|cuda:N`, optional `torch.compile` for the CUDA `fast` recipe (needs Triton), folder or
    `--recursive` input and PNG output, with the detected architecture/scale/dtypes/recipe/tiling printed up front
- `nmanga torch to-onnx` - The ONNX export moved under the `torch` group, with the same options as before
- The ONNX session builders, engine cache, tiling, ORT initialization and Windows ML discovery moved into a new
  `nmanga.ml` package. `nmanga.denoiser`, `nmanga.winml` and `nmanga.lazy.get_onnxruntime` stay importable as
  thin forwards over it, and share one ONNX Runtime instance and initialization state
- `safetensors` is an explicit `denoiser` dependency now, since the PyTorch backend reads checkpoint headers
  directly to report the stored dtypes
- `nmanga orchestra` - Create/run a JSON file that will run multiple functions in nmanga together
- `nmanga orchestra run-target` - Run the whole action chain on a single target directory, without having to
  list it in the configuration `volumes`
  - The target has to live inside the configured `base_path`, and is turned into an ephemeral volume using
    `--volume`/`-vol` (supports omnibus ranges like `1-2`), or treated as a oneshot when no volume is given
  - Actions that need volume metadata (`rename`, `spreads`) are refused before anything runs, since the
    action list is shared with `orchestra run`
  - `--dry-run` prints what every action would do without touching any file
- Orchestrator volume `colors` - Accept `null` to mark the color pages as unknown, so color actions detect
  them on the fly instead of treating the volume as having no color pages at all
- `nmanga lookup imagesize` - Lookup image sizes in an archive or folder
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Add `-ws/--workspace-size`,
  `-cg/--cuda-graph` and `-sep/--strict-ep` so a TensorRT comparison can reproduce the workspace limit, the
  CUDA graph setting and refuse a silent fallback to another execution provider
  - `nmanga denoise-trt` also grows `-rtx/--with-nvrtx` and `-qs/--quant-size`; passing any of the new options
    routes it through the TensorRT RTX builder instead of the legacy loader that forces FP16 onto the whole graph
- Orchestrator `denoise`/`upscale` actions - Add `workspace_size`, `enable_cuda_graph` and `strict_ep` fields
- Orchestrator `autolevel` action - The `v3` algorithm is now actually implemented, backed by the `nimages`
  VapourSynth plugin like `nmanga autolevel3`. It builds one clip per volume and levels the pages with the same
  per page decisions as `v2`; colour pages are levelled on their own planes with the levels found on their luma
  - The action now carries the `cache_mb` and `prefetch` options the command has
- Orchestrator `posterize` action - The `vapoursynth` mode is now actually implemented, like `nmanga posterize2`.
  The SSIMULACRA2 gate compares against the gray clip the posterize chain already decoded, and colour pages are
  copied as before
  - The action now carries the `cache_mb` and `prefetch` options the command has
- `nmanga posterize2` - Metric guided posterization. `--bits` now takes one depth or several, and several turn
  the command into a fan-out over the `vship` metrics
  - `--bits 2-5`, `--bits 2,3,4` or `--bits 1-6` posterize every candidate depth for a chunk of pages, score
    them in one `vship` call per candidate per metric, and keep the least degraded depth every thresholded
    metric accepts
  - A single depth with `--metrics` is the same rule with one candidate: the page is scored and copied when
    it fails, so `--bits 4 --metrics ssimulacra2` is the old single depth gate
  - A page no candidate passes is copied from the source, so a degraded page never reaches the output
  - `--metrics` picks what scores the candidates (`ssimulacra2`, `butteraugli`, `cvvdp`), `--metric-min` sets a
    threshold per metric, and `--bits-policy` chooses between "fewest bits that passes" and the order given
  - Each metric's direction is fixed and asserted against a known anchor: identical input scores `100` for
    `ssimulacra2`, `0` for `butteraugli`, and `10` for `cvvdp` with `standard_4k`, so the banner line reads
    `cvvdp>=9.9` rather than `<=`
  - One depth with no `--metrics` scores nothing, so `--bits 4` stays the fixed depth posterize it always was
  - Each candidate hangs off one shared decode, so a volume costs one decode, one posterize pass per candidate,
    and one metric dispatch per candidate per metric rather than one per page
  - Pages of differing sizes are grouped per size, because `vship` takes one size per call
  - `--chunk` (default 16) sizes how many pages are scored and written together, against the frame cache
  - **Breaking:** `--use-ssimulacra2` and `--ssim-min` are gone. Their equivalent is `--bits <depth> --metrics
    ssimulacra2`, which scores that one depth at the same `80` default and copies the pages that fail
- Reworked CLI display with `rich-click` for better experience
- Implemented threaded tagging for better performance
- Use `rich` progress bar for better progress display
- Implement better support for outputing to STDOUT/STDERR in threaded functions
- Add `--recursive` option to the following commands:
  - `nmanga denoise-trt`
  - `nmanga autolevel2`
  - `nmanga jpegify`
- `nmanga-gui` - Optional PySide6 GUI for the manual split feature, install with the `gui` extra
  - Load a folder of images or an archive and see the pages as a thumbnail grid
  - Drag a folder or archive from the file manager and drop it anywhere on the window to load it
  - Drag-select pages into chapter ranges, type the ranges directly, or press Add to append the next range
  - Skip pages with a comma separated list such as `1,5-20`, which is kept visible in the range and in the
    uncovered page warnings
  - Preview which chapter owns each page, and get warned about uncovered, overlapping, or missing pages,
    with the page numbers listed
  - Carry a volume per chapter, so an omnibus can be split into `v01-02` chapters
- `nmanga autolevel3`, `nmanga posterize2`, `nmanga deblur2` - The PNG encode and the publish now belong to
  `imgseqs.PNGWrite` instead of Pillow on worker threads
  - One writer node covers a whole clip and names each still from the page's own source, so the three
    commands write the destination directly: no staging directory and no rename anywhere
  - `--threads` is now the number of open PNG write requests. End to end this is 1.1x to 1.3x over the
    six worker Pillow route at `-t 6` on 1404x2000 and 3672x5274 volumes, with decoded pixels identical
    and files within 1% of the old size
  - `nmanga deblur2` no longer fails on a 16-bit source: the writer stores a 16-bit PNG instead of the old
    Pillow path raising `Cannot handle this data type`
  - The writer publishes truthful PNG metadata (an `sBIT` for a widened 9-15 bit frame, a `cICP` when the
    source states primaries and transfer), so a whole file hash can change even when the pixels do not
- `nmanga autolevel3` - One output mode replaces `--keep-colorspace` and `--force-gray`
  - `--color/--gray`, colour by default: a page keeps its own container's colour, a YUV page is converted
    to RGB by the writer with the frame's own matrix and range, and `--gray` writes one plane per page, a
    skipped page included
  - The analyzer reads the source pages themselves: `nimages` computes an `RGB24` page's luma (Pillow's
    `convert("L")` exactly) and takes plane 0 of a gray or YUV page, so no normalisation pass runs first
  - A skipped page is copied byte for byte when it is already in the output mode and re-encoded from the
    un-levelled analysed page when it is not
  - `--prefetch` (default 16, `0` disables) sizes the decoder lookahead, `--cache` bounds the VapourSynth
    frame cache, and `-v` reports the plugin's resolved arguments, per frame timings, every page's levels
    and its peak-found flags
- `nmanga autolevel3` and `nmanga posterize2` - `--gpu` runs the analysis and the leveling on the device
  - Off by default: with a host decoder and a host writer the chain pays an upload and a download per page
  - Needs `vapoursynth-nimages` 0.2.1 or newer and a Vulkan 1.4 device, and fails with the plugin's own
    message instead of falling back to the host
- `nmanga posterize2` - Output is written from the chosen `Posterize` output instead of a float round
  trip, and a volume whose pages resolve to one destination (such as `page.jpg` beside `page.png`) is
  refused before the first write
  - `-v` reports the plugin's resolved arguments, per frame timings and the per page scores

**Changes**
- `nmanga autolevel3` - **Breaking:** `--keep-colorspace`, `--force-gray` and `-f/--format` are gone. Output
  is PNG only, and the default flips from gray to colour: pass `--gray` for the old default behaviour
- Improve manual split filename matcher
- `nmanga manualsplit` now accepts a folder of images, not just an archive
- `nmanga manualsplit` supports a volume per chapter range (including omnibus ranges such as `1-2`)
  - Use `-vol 1-2` for a whole-omnibus default, or `--per-chapter-volume` to set one per chapter
- `nmanga manualsplit` accepts a comma separated list of pages (for example `1,5-20`) to skip pages inside a
  chapter range; a comma turns off the bare number "until the end" form so the last page must be given
- Move denoise and spreads joiner to the main nmanga module instead of in CLI part only
- Adjust image gray level peak detection
- Fix some issues with PDF images extraction and compositing
- Allow modifying compression level when packing archives
  - Use `-cl/--compression-level` option in `nmanga pack` and `nmanga packepub`
  - Alternatively, use `compress_level` in the `pack` orchestrator action
- Move from standard `click` to `rich-click` for better CLI experience
- Make threaded worker run function serially when threads is set to 1 to avoid overhead
- Split the machine learning extras by hardware vendor. `denoiser` keeps the vendor neutral stack (torch,
  ONNX, einops), and one of the `denoiser-*` extras adds the execution provider
  - `denoiser-ort` - the stock `onnxruntime` wheel: CoreML on macOS and a CPU provider everywhere else.
    This is the runtime the plugin EP extras load into
  - `denoiser-nvidia` - `onnxruntime-gpu`, the TensorRT RTX plugin and `tensorrt-cu13`, with the `+cu132`
    torch wheel. It replaces the stock runtime instead of extending it, so it is declared as conflicting
    with `denoiser-ort` and `denoiser-mlx` and `uv sync --all-extras` is now an error; CI installs
    `denoiser-ort`
  - `denoiser-intel` - the `onnxruntime-ep-openvino` plugin next to either runtime, so it can be combined
    with `denoiser-ort` or `denoiser-nvidia`. It works on 3.14, unlike the old `onnxruntime-openvino` fork
    it replaces
  - `denoiser-mlx` - the `onnxruntime-ep-mlx` plugin for Apple Silicon (macOS 14+)
  - There is no `denoiser-amd` extra yet: support is expected to arrive through Windows ML instead
  - The out-of-tree plugin EPs (TensorRT RTX, MLX, OpenVINO) are registered with ONNX Runtime in the shared
    loader, before any session or engine builder exists
  - OpenVINO picks the CPU/GPU/NPU device the plugin exposes instead of the in-tree `HETERO:GPU,CPU` option
  - With an accelerator already ahead of it (CUDA or TensorRT RTX) OpenVINO is added as a name only
    fallback instead of a provider device, so asking for both keeps CUDA as the primary provider
- `numpy` and `scipy` are regular dependencies now, so the `autostuff` extra is gone and the machine
  learning extras no longer have to restate `numpy`

**Fixes**
- Orchestrator `posterize` action - `bpc: auto` could not be used at all: the `ge`/`le` range on the field was
  applied to the `auto` literal too, so constructing the action raised a `TypeError` from inside pydantic. The
  range is now checked by a validator that leaves `auto` alone
- `nmanga shiftname` and the orchestrator `shift_rename` action - Work out the order to rename the files in
  automatically, so no file is overwritten by another, and fail instead of guessing when no safe order exists
  - A right shift is renamed descending and a left shift ascending, derived from the mapping itself; there is no
    option to configure, and `-r/--reverse` keeps meaning "number the pages bottom up"
  - The whole plan is validated before the first rename, so a mapping that cannot be done safely (a cycle such
    as `--reverse` over already numbered pages, two files wanting one name, or a stray file sitting on a
    destination) leaves the folder untouched
  - A rename that fails partway is rolled back in reverse, which always restores the original names; if an undo
    fails too, the error names the files that are still displaced instead of reporting a clean abort
  - Refuses to replace an existing file during the move, atomically via `renameat2(RENAME_NOREPLACE)` on Linux
    and `renamex_np(RENAME_EXCL)` on macOS, with a portable existence check as the fallback
  - Add `-n/--dry-run` to print the order the files would be renamed in
- `nmanga shiftname` - Report a non-zero exit code when a rename cannot be performed. The command previously
  returned a value that click discards, so a failure still exited 0
- `nmanga upscale-trt` and `nmanga denoise-trt` - Clamp the batch size down to the one hardcoded in the model
  (e.g. an input shape of `[1, 3, height, width]`) instead of failing with an `INVALID_ARGUMENT` error
- `nmanga upscale-trt` and `nmanga denoise-trt` - Report a proper error when the model declares a fixed tile size
  that differs from the requested one
- `nmanga-gui` - Check the chapter ranges against the images of the source, so a range ending on the second half
  of a spread is no longer reported as referencing a page missing from the source
- `nmanga-gui` - Forget the loaded pages and chapter ranges when a source fails to load, instead of silently
  keeping the previous source
- `nmanga-gui` - Spell out the skipped pages of a hand picked chapter range (for example `001, 005-007`)
  instead of collapsing them into one start-to-end label
- `nmanga upscale-trt` and `nmanga denoise-trt` - Build the TensorRT optimization profiles from the dimensions
  the model declares, so a fixed `512x512` input no longer gets a contradictory default profile that only failed
  at the first inference. A fixed batch size is adopted and a mismatching fixed tile size is rejected up-front
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Key the compiled engine cache (and
  the in-memory session cache) on the model contents, the precision, the workspace, the CUDA graph setting and the
  TF32 policy, instead of only the model pathname and runtime version. Changing the weights in place, or switching
  TF32 off, now rebuilds the engine instead of reusing an unsuitable one
- ONNX Runtime telemetry is disabled in the shared loader, before any session or engine builder exists
- `nmanga upscale-trt`, `nmanga denoise-trt` and the orchestrator ML actions - Probe the model scale factor with the
  batch size the engine was built for, so a fixed-batch graph is not probed with a batch it does not accept
- Fix the TRT-RTX GPU capability check to query the requested device instead of the current one
- The GPU memory probe no longer raises when torch has no usable CUDA device (CPU-only wheels, or a CUDA
  build on a machine without the NVIDIA driver), so the AMD/Intel extras fall back to the default workspace
