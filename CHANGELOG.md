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
- Reworked CLI display with `rich-click` for better experience
- Implemented threaded tagging for better performance
- Use `rich` progress bar for better progress display
- Implement better support for outputing to STDOUT/STDERR in threaded functions
- Add `--recursive` option to `nmanga denoise-trt` and `nmanga autolevel2` command to process all directories inside a given folder

**Changes**
- Improve manual split filename matcher
- Move denoise and spreads joiner to the main nmanga module instead of in CLI part only
- Adjust image gray level peak detection
- Fix some issues with PDF images extraction and compositing
- Allow modifying compression level when packing archives
  - Use `-cl/--compression-level` option in `nmanga pack` and `nmanga packepub`
  - Alternatively, use `compress_level` in the `pack` orchestrator action
- Move from standard `click` to `rich-click` for better CLI experience
- Make threaded worker run function serially when threads is set to 1 to avoid overhead
