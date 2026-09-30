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
- `nmanga orchestra` - Create/run a JSON file that will run multiple functions in nmanga together
- `nmanga lookup imagesize` - Lookup image sizes in an archive or folder
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

**Changes**
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
- `nmanga autolevel3` - (Experimental) automatic color leveling backed by the `nimages` VapourSynth plugin
  - One clip is built per directory, and the pages are pulled in order while `--threads` pages are encoded
    and written at once. The PNG encode is roughly 80% of the per page cost and Pillow releases the GIL for
    it, so this is where the parallelism pays: 129 1404x2000 jpeg pages go from ~15 to ~68 pages per second
    at `-t 8`, with byte identical output
  - `--prefetch` (default 16, `0` disables) sizes the decoder lookahead; `--cache` bounds the VapourSynth
    frame cache
  - Same per page decisions as `autolevel2` (`--keep-colorspace`, `--force-gray`, `--no-white`,
    `--peak-offset`, `-f/--format`, `-r/--recursive`), minus `--legacy` and `--use-magick` which have no
    plugin equivalent
  - Pass `-v` for the plugin's resolved arguments, per frame stage timings, and the black/white level and
    peak-found flags of every page
  - Unlike `autolevel2 --format jpg`, which raises a `KeyError` because Pillow has no `JPG` writer, this
    command writes a real JPEG
- `nmanga posterize2` - (Experimental) posterize to a fixed bit depth backed by the `nimages` VapourSynth plugin
  - Same `--bits`, `--use-ssimulacra2`, `--ssim-min`, `--prefetch`, `--threads` and `-r/--recursive` options
    as `autolevel3`; `--threads` drives the same write pool, so 129 pages go from ~21 to ~77 per second at
    `-t 6` without the gate
  - The gate itself is a single GPU filter and stays on the main thread
  - The SSIMULACRA2 gate compares the posterized page against the gray clip the posterize chain already
    decoded, so the second pull costs no extra decode
  - Pass `-v` for the plugin's resolved arguments and per frame stage timings
  - Output is byte identical to `posterize` on lossless input; on lossy input the two decoders differ by one
    code value on a small share of pixels, so a byte comparison is only valid on png
