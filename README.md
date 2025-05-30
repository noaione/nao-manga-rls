# [nao] Manga Release Scripts

This repo contains stuff that I use to release and collect manga from a certain cat website.

All my release use one of the following tag:
- `[nao]`, default release tag for any of my stuff
- `(TooManyIsekai)` any release that has Isekai element in it. (Normal Fantasy does not count)
- `(oan)`, new tag for lower effort stuff that I use.
- `(naoX)`, tag for upscaled stuff that I did.

## Requirements
- Python 3.10+
- imagemagick (for `spreads join` and `spreads split` command)
- exiftool (for `releases` and `tag`, optional)
- pingo (for `releases` and `optimize`, optional)

## Scripts
This repo also have a module or script to release and split stuff up.

To install, you need git since I'm not publishing this to PyPi.
After that you can run this:

```sh
pip install -U git+https://github.com/noaione/nao-manga-rls.git
```

This will install this project which then you can use the command `nmanga` to execute everything.

## Configuration

This project or script use `nmanga config` command to configure defaults and executables path.

You can run the command to configure some stuff, you will get warned if you haven't done the first setup yet.

You can configure the following:
- Defaults
  - Bracket type (round, square, curly)
  - Ripper credit name (used in the filename)
  - Ripper credit email (used in the `.cbz` file comment and image tagging at author/comment)
  - Add `c` prefix to chapter name when packing. (ex: `Title c001 (20xx)` instead of `Title 001 (20xx)`)
  - Use `#` as chapter special separator, if your name is `1r0n` enable this. (ex: `Title - c001#1 (vXX) ----` instead of `Title - c001x1 (vXX) ----`)
  - Publication type for `releases` command (Default: `digital`)
    - If you don't want the `[dig]` thing to be added, you can remove it by selecting `none` type.
  - Publication type for `releasesch` command (Default: `digital`)
    - If you don't want the `[dig]` thing to be added, you can remove it by selecting `none` type.
- Executables path
  - `pingo`
  - `magick`/`imagemagick`
  - `exiftool`

The configuration file will be saved in `.json` format in the following path:
- Windows: `%APPDATA\nmanga\config.json`
- Linux/macOS: `~/.config/nmanga/config.json`

## Changelog

See here: [CHANGELOG.md](https://github.com/noaione/nao-manga-rls/blob/master/CHANGELOG.md)

### Commands

#### `autosplit`
Same as the old auto split script, this command accept some parameter.

```py
Usage: nmanga autosplit [OPTIONS] PATH_OR_ARCHIVE_FILE

  Automatically split volumes into chapters using regex

Options:
  -t, --title TEXT         The title of the series (used on volume filename,
                           and inner filename)  [required]
  -pub, --publisher TEXT   The publisher of the series (used on inner
                           filename)
  -it, --inner-title TEXT  The title of the series (used on inner filename,
                           will override --title)
  -lt, --limit-to TEXT     Limit the volume regex to certain ripper only.
  -oshot, --is-oneshot     Mark the series as oneshot
  -h, --help               Show this message and exit.
```

`--title` will basically try to match with the volume filename and the image filename.<br />
`--inner-title` will override the image filename matching pattern.<br />
`--publisher` Only use this if you want to add chapter title and there is actual chapter title.<br />
`--is-oneshot` mark it as one shot (no volume information basically)<br />
`--limit-to` Limit the regex to certain group/ripper.

The regex should match something like this:
- `Manga Title - c001 (v01) - p001 [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg` (`danke/nao`)
- `Manga Title - c001 (v01) - p000 [Cover] [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg` (`danke/nao`)
- `Manga Title - c001 (v01) - p000 [Cover] [dig] [Publisher Name?] [Group] {HQ}.jpg` (`danke/LuCaZ/nao`)
- `Manga Title - c001 (v01) - p001 [dig] [Publisher Name?] [Group] {HQ}.jpg` (`danke/LuCaZ/nao`)
- `Manga Title - c001 (v01) - p001 [Publisher Name?] [Group] {HQ}.jpg` (`1r0n`)
- `Manga Title - c001x1 (c001.1) (v01) - p001 [dig] [Publisher Name?] [Group] {HQ}.jpg`

The `{HQ}` thing is optional, for the last one you cannot add chapter name to it.

The output format would be like this:
- If `[Chapter Title]` exist and `-pub/--publisher` are used: `01.001 - Chapter Title.cbz`
- If `[Chapter Title]` **DOES NOT** exist and `-pub/--publisher` are used: `01.001.cbz`
- If `-pub/--publisher` are **NOT** used: `01.001.cbz`
- If `--is-oneshot` are used, **sometimes**: `00.001.cbz`
- If there is `#1` or `x1` or similar after the chapter number, it would be: `01.001.5.cbz` (or `01.001.5 - Chapter Title.cbz` with chapter title)
- If there is something like `(c001.1)` after the chapter number: `01.001.1.cbz` (or `01.001.1 - Chapter Title.cbz` with chapter title)

You can play around and see what it would result with, report any problem in the [Issues](https://github.com/noaione/nao-manga-rls/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc) if there is unexpected results.

This should match most of the release from a certain cat website (`danke`, `LuCaZ`, `1r0n`, `nao`, etc.)

#### ~~`level`~~

**Deprecated/Removed!**

Use `magick mogrify` instead.

Example:
- `magick mogrify -format png -alpha off -colorspace Gray -level 12.75%,100% -path ./output/ ./target/*.jpg`

#### `manualsplit`
Same as the old manual split script, this command accept some parameter and will then ask a series of question.

```py
Usage: nmanga manualsplit [OPTIONS] ARCHIVE_FILE

  Manually split volumes into chapters using multiple modes

Options:
  -vol, --volume INTEGER  The volume number for the archive
  -h, --help              Show this message and exit.
```

`--volume` will mark the archive file with the proper volume number.

After that, you should be asked on what mode you will be using:
```py
[?] Select mode: Page number mode (all filename must be page number)
 > Page number mode (all filename must be page number)
   Regex mode (Enter regex that should atleast match the page number!)
   Page number mode with custom page number mapping
   Regex mode with custom page number mapping
```

- `Page number mode`, use if the image filename only contains page number (`000.jpg`, etc)
- `Regex mode`, use if you need to use custom regex to match the page number (follow the default example.) (`i-000.jpg`, etc.)

Custom page number mapping will ask you some custom page mapping, for example if you have `cover.jpg` you can use this mode then map that to page 0.<br />
After that you will be asked a series of question regarding the chapter mapping.

#### `merge`
Merge two or more archives file into a single archive file.

```py
Usage: nmanga merge [OPTIONS] ARCHIVE_FILES

  Merge chapters together into a single chapter

Options:
  -o, --output TEXT  Override the output file, will default to first input if
                     not provided!
  -h, --help         Show this message and exit.
```

`-output` the output filename, if not provided will use the name of the first archive.

Example:
- `nmanga merge a.cbz b.cbz` will merge into `a.cbz`
- `nmanga merge -o test.cbz a.cbz b.cbz` will merge into `test.cbz`

The `ARCHIVE_FILES` input can be repeated as many times as you want!

**Note**<br />
The archive file will be deleted in favor of the new one!

#### `optimize`
Optimize images with pingo

```py
Usage: nmanga optimize [OPTIONS] FOLDER_PATH

  Optimize images with pingo

Options:
  -ax, --aggressive BOOLEAN  [default: False]
  -pe, --pingo-exec TEXT     Path to the pingo executable  [default: pingo]
  -h, --help                 Show this message and exit.
```

`--aggressive`, use `-jpgtype=1` for JPEG.<br />
`--pingo-exec`, the path to pingo executable.

It will automatically detect for the relevant images and apply "opinionated" optimization.

- JPG: `-s0 -strip` (also include `-jpgtype=1` if `--aggressive`)
- PNG: `-sb -strip`
- WEBP: `-s9 -strip`

It's recommended to run this before tagging the image!

#### `pack`
Pack a release to a `.cbz` archive.

```py
Usage: nmanga pack [OPTIONS] FOLDER_PATH

  Pack a release to an archive.

Options:
  -t, --title TEXT                The title of the series  [required]
  -y, --year INTEGER              The year of the series release
  -vol, --volume INT_OR_FLOAT     The volume of the series release
  -ch, --chapter INT_OR_FLOAT     The chapter of the series release
  -pt, --publication-type [digital|magazine|scan|web|digital-raw|magazine-raw|mix|none]
                                  The publication type for this series, use
                                  none to remove it from image filename
                                  [default: digital]
  -c, --credit TEXT               The ripper credit for this series  [default:
                                  nao]
  -e, --email TEXT                The ripper email for this series  [default:
                                  noaione@protonmail.ch]
  -r, --revision INTEGER RANGE    The revision of the release, if the number 1
                                  provided it will not put in the filename
                                  [default: 1; x>=1]
  -ex, --extra-meta TEXT          Extra metadata to add to the pack filename
  -br, --bracket-type [square|round|curly]
                                  Bracket to use to surround the ripper name
                                  [default: square]
  -m, --mode [folder|cbz|cb7|epub]
                                  The output mode for the archive packing
                                  [default: ExporterType.cbz]
  -h, --help                      Show this message and exit.
```

`--title`, the series title<br />
`--year`, the series year (will be used for exif tagging)<br />
`--volume`, the volume number<br />
`--chapter`, the chapter number<br />
`--publication-type`, the publication type (`digital`, `magazine`, `scan`, `web`, `digital-raw`, `magazine-raw`, or `mix`)<br />
`--credit`, the ripped/group name<br />
`--email`, will be used for exif tagging<br />
`--revision`, the revision number of the releases.<br />
`--extra-meta`, used in the archive filename before the publication type<br />
`--bracket-type`, the bracket to be used.<br />
`--mode`, the output for archive output, `folder` will be ignored in this case.

This will zipped all of the images in a folder.
And will also add your email to the archive comment.

For `--volume` and `--chapter`, provide one of them and the packed zip will be a bit different.
- `--volume`: `Manga Title vXX (20xx) (Digital) (XXX)`
- `--chapter`: `Manga Title XXX (20xx) (Digital) (XXX)`

If you provide both, `--volume` will take priority.

For `--publication-type`, if you don't want the `[dig]` thing to be added, you can remove it by selecting `none` type.

For `--extra-meta`, this will add before any publication type so it will be like this:
- `Manga Title vXX (20xx) (EXTRA META HERE) (Digital) (XXX)`

#### `releases`
Create a release from `comix` formatted filename.

```py
Usage: nmanga releases [OPTIONS] FOLDER_PATH

  Prepare a release of a manga series.

Options:
  -t, --title TEXT                The title of the series  [required]
  -y, --year INTEGER              The year of the series release
  -pub, --publisher TEXT          The publisher of the series  [required]
  -pt, --publication-type [digital|magazine|scan|web|digital-raw|magazine-raw|mix|none]
                                  The publication type for this series, use
                                  none to remove it from image filename
                                  [default: digital]
  -c, --credit TEXT               The ripper credit for this series  [default:
                                  nao]
  -e, --email TEXT                The ripper email for this series  [default:
                                  noaione@protonmail.ch]
  -r, --revision INTEGER RANGE    The revision of the release, if the number 1
                                  provided it will not put in the filename
                                  [default: 1; x>=1]
  -ex, --extra-meta TEXT          Extra metadata to add to the pack filename
  -hq, --is-high-quality          (DEPRECATED) Whether this is a high quality
                                  release
  -mq, --quality [LQ|HQ]          Image quality of this release.
  --tag / --no-tag                Do exif metadata tagging on the files.
                                  [default: tag]
  --optimize / --no-optimize      Optimize the images using pingo.  [default:
                                  no-optimize]
  -ee, --exiftool-exec TEXT       Path to the exiftool executable  [default:
                                  exiftool]
  -pe, --pingo-exec TEXT          Path to the pingo executable  [default:
                                  pingo]
  -br, --bracket-type [square|round|curly]
                                  Bracket to use to surround the ripper name
                                  [default: square]
  -h, --help                      Show this message and exit.
```

`--title`, the series title<br />
`--year`, the series year (will be used for exif tagging)<br />
`--publisher`, the publisher<br />
`--publication-type`, the publication type (`digital`, `magazine`, `scan`, `web`, `digital-raw`, `magazine-raw`, or `mix`)<br />
`--credit`, the ripped/group name<br />
`--email`, will be used for exif tagging<br />
`--revision`, the revision number of the releases.<br />
`--extra-meta`, used in the archive filename before the publication type<br />
**[DEPRECATED]** `--is-high-quality`, mark the release as HQ (add `{HQ}` to filename)<br />
`--quality`, the image quality of this release (optional, add `{HQ}` or `{LQ}` to filename)<br />
`--tag/--no-tag`, do exif tagging.<br />
`--optimize/--no-optimize`, optimize image with pingo<br />
`--bracket-type`, the bracket to be used.

After you mark the folder or archive, you will be asked a series of question.
You can enter all the information and after that the program will rename everything and tag according to your specification.

This use the same inquiring method as `manualsplit`

The filename also must match something like this:
- `Manga Title - vXX - pXXX`

For `--publication-type`, if you don't want the `[dig]` thing to be added, you can remove it by selecting `none` type.

For `--extra-meta`, this will add before any publication type so it will be like this:
- `Manga Title vXX (20xx) (EXTRA META HERE) (Digital) (XXX)`

#### `releasesch`
Create a release for a single chapter, this will format the filename into the formatting we wanted.

```py
Usage: nmanga releasesch [OPTIONS] FOLDER_PATH

  Prepare a release of a manga chapter.

Options:
  -t, --title TEXT                The title of the series  [required]
  -y, --year INTEGER              The year of the series release
  -pub, --publisher TEXT          The publisher of the series  [required]
  -ch, --chapter INT_OR_FLOAT     The chapter of the series release
  -vol, --volume INTEGER          The volume of the series release
  -cht, --chapter-title TEXT      Chapter title that will be included between
                                  the publication type and publisher
  -pt, --publication-type [digital|magazine|scan|web|digital-raw|magazine-raw|mix|none]
                                  The publication type for this series, use
                                  none to remove it from image filename
                                  [default: web]
  -c, --credit TEXT               The ripper credit for this series  [default:
                                  nao]
  -e, --email TEXT                The ripper email for this series  [default:
                                  noaione@protonmail.ch]
  -r, --revision INTEGER RANGE    The revision of the release, if the number 1
                                  provided it will not put in the filename
                                  [default: 1; x>=1]
  -hq, --is-high-quality          (DEPRECATED) Whether this is a high quality
                                  release
  -mq, --quality [LQ|HQ]          Image quality of this release.
  --tag / --no-tag                Do exif metadata tagging on the files.
                                  [default: tag]
  --optimize / --no-optimize      Optimize the images using pingo.  [default:
                                  no-optimize]
  -ee, --exiftool-exec TEXT       Path to the exiftool executable  [default:
                                  exiftool]
  -pe, --pingo-exec TEXT          Path to the pingo executable  [default:
                                  pingo]
  -br, --bracket-type [square|round|curly]
                                  Bracket to use to surround the ripper name
                                  [default: square]
  -h, --help                      Show this message and exit.
```

`--title`, the series title<br />
`--year`, the series year (will be used for exif tagging)<br />
`--publisher`, the publisher<br />
`--chapter`, the chapter number<br />
`--chapter-title`, the chapter title, same formatting if you use with chapter title mode on `releases`<br />
`--volume`, the chapter associated volume, if not provided it will use `(NA)`<br />
`--publication-type`, the publication type (`digital`, `magazine`, `scan`, `web`, `digital-raw`, `magazine-raw`, or `mix`)<br />
`--credit`, the ripped/group name<br />
`--email`, will be used for exif tagging<br />
`--revision`, the revision number of the releases.<br />
**[DEPRECATED]** `--is-high-quality`, mark the release as HQ (add `{HQ}` to filename)<br />
`--quality`, the image quality of this release (optional, add `{HQ}` or `{LQ}` to filename)<br />
`--tag/--no-tag`, do exif tagging.<br />
`--optimize/--no-optimize`, optimize image with pingo<br />
`--bracket-type`, the bracket to be used.

This will automatically format everything depending on the option provided.<br/>
For example:
- Command: `nmanga releasesch -t "The Necromancer Maid" -pub "Comikey" -ch 26 -pt web -cht "Princess Chloe" ./c026`
  - Output: `The Necromancer Maid - c026 (NA) - p000 [web] [Princess Chloe] [Comikey] [RIPPER]`
- Command: `nmanga releasesch -t "The Necromancer Maid" -pub "Comikey" -vol 3 -ch 26 -pt web -cht "Princess Chloe" ./c026`
  - Output: `The Necromancer Maid - c026 (v03) - p000 [web] [Princess Chloe] [Comikey] [RIPPER]`

The filename must have a minimum match like this:
- `pXXX`
- `pXXX-YYY`

Anything outside that is ignored.

For `--publication-type`, if you don't want the `[dig]` thing to be added, you can remove it by selecting `none` type.

#### `spreads`
Manage spreads from a directory of images.

```py
Usage: nmanga spreads [OPTIONS] COMMAND [ARGS]...

  Manage spreads from a directory of images

Options:
  -h, --help  Show this message and exit.

Commands:
  join   Join multiple spreads into a single image
  split  Split a joined spreads into two images
```

##### `spreads join`
Join multiple spreads into a single image.

```py
Usage: nmanga spreads join [OPTIONS] FOLDER_PATH

  Join multiple spreads into a single image

Options:
  -q, --quality FLOAT RANGE    The quality of the output image  [default:
                               100.0; 1.0<=x<=100.0]
  -s, --spreads A-B            The spread information, can be repeated and
                               must contain something like: 1-2  [required]
  -r, --reverse                Reverse the order of the spreads (manga mode)        
  -f, --format [auto|png|jpg]  The format of the output image, auto will
                               detect the format from the input images
                               [default: auto]
  -me, --magick-exec TEXT      Path to the magick executable  [default:
                               magick]
  -h, --help                   Show this message and exit.
```

`--quality`, the output quality (mainly used as jpg export)<br />
`--spreads`, the spread information, can be repeated (ex: `-s 3-4 -s 99-100` will merge page 3 and 4 together.)<br />
`--reverse`, reverse the order of the spread (recommended for manga/RTL layout)<br />
`--format`, the output format that should be used, default to `auto` that will determine the output format from the input images.

The filename should have the minimum format like this: `pXXX`<br />
The prefix `p` is important to differentiate it from any other text in the filename.<br />
Everything else is ignored and will be included on the final filename.

This will also make a `backup` folder which contains the unmerged images (in case something went wrong.)

##### `spreads split`
Split a joined spreads into two images.

```py
Usage: nmanga spreads split [OPTIONS] FOLDER_PATH

  Split a joined spreads into two images

Options:
  -q, --quality FLOAT RANGE    The quality of the output image  [default:
                               100.0; 1.0<=x<=100.0]
  -r, --reverse                Reverse the order of the spreads (manga mode)        
  -f, --format [auto|png|jpg]  The format of the output image, auto will
                               detect the format from the input images
                               [default: auto]
  -me, --magick-exec TEXT      Path to the magick executable  [default:
                               magick]
  -h, --help                   Show this message and exit.
```

`--quality`, the output quality (mainly used as jpg export)<br />
`--reverse`, reverse the order of the spread (recommended for manga/RTL layout)<br />
`--format`, the output format that should be used, default to `auto` that will determine the output format from the input image.

The filename should have the minimum format like this: `pXXX-YYY`<br />
The prefix `p` is important to differentiate it from any other text in the filename.<br />
Everything else is ignored and will be included on the final filename.

Same with `spreads join`, a `backup` folder will be created which contains the merged image (in case something went wrong.)

#### `tag`
Tag images with exif metadata, only works for `.tiff` and `.jpg` files!

```py
Usage: nmanga tag [OPTIONS] FOLDER_PATH

  Tag images with metadata

Options:
  -t, --title TEXT                The title of the series  [required]
  -vol, --volume INT_OR_FLOAT     The volume of the series release
  -ch, --chapter INT_OR_FLOAT     The chapter of the series release
  -y, --year INTEGER              The year of the series release
  -pt, --publication-type [digital|magazine|scan|web|digital-raw|magazine-raw|mix|none]
                                  The publication type for this series, use
                                  none to remove it from image filename
                                  [default: digital]
  -c, --credit TEXT               The ripper credit for this series  [default:
                                  nao]
  -e, --email TEXT                The ripper email for this series  [default:
                                  noaione@protonmail.ch]
  -r, --revision INTEGER RANGE    The revision of the release, if the number 1
                                  provided it will not put in the filename
                                  [default: 1; x>=1]
  -ex, --extra-meta TEXT          Extra metadata to add to the pack filename
  -br, --bracket-type [square|round|curly]
                                  Bracket to use to surround the ripper name
                                  [default: square]
  -ee, --exiftool-exec TEXT       Path to the exiftool executable  [default:
                                  exiftool]
  -h, --help                      Show this message and exit.
```

`--title`, the series title<br />
`--year`, the series year (will be used for exif tagging)<br />
`--volume`, the volume number<br />
`--chapter`, the chapter number<br />
`--publication-type`, the publication type (`digital`, `magazine`, `scan`, `web`, `digital-raw`, `magazine-raw`, or `mix`)<br />
`--credit`, the ripped/group name<br />
`--email`, will be used for exif tagging<br />
`--revision`, the revision number of the releases.<br />
`--extra-meta`, used in the archive filename before the publication type<br />
`--bracket-type`, the bracket to be used.

This will automatically find any valid images that can be tagged with exif metadata and apply it!

For `--volume` and `--chapter`, provide one of them and the packed zip will be a bit different.
- `--volume`: `Manga Title vXX (20xx) (Digital) (XXX)`
- `--chapter`: `Manga Title XXX (20xx) (Digital) (XXX)`

For `--publication-type`, if you don't want the `[dig]` thing to be added, you can remove it by selecting `none` type.

For `--extra-meta`, this will add before any publication type so it will be like this:
- `Manga Title vXX (20xx) (EXTRA META HERE) (Digital) (XXX)`

## API Usage

You can use this module as an API too, if you want to follow what the CLI doing you can view the code and implement your own version.

For example, optimizing images in a folder:

```py
from pathlib import Path

from nmanga.common import optimize_images

target_dir = Path("target")
pingo_path = "pingo"

optimize_images(pingo_path, target_dir)
# Or aggresive: optimize_images(pingo_path, target_dir, True)
```

Creating Daiz-like filename formatting for archive/image filename

```py
from nmanga.common import ChapterRange, format_daiz_like_filename

filename, archive_name = format_daiz_like_filename(
    manga_title="Manga Title",
    manga_publisher="Real Publisher",
    manga_year=2023,
    chapter_info=ChapterRange(1, "Chapter 1", [0]),
    page_number="001",
    publication_type="digital",
    ripper_credit="nao",
    bracket_type="round",
    manga_volume="v01",
    extra_metadata="Cover",
    image_quality="HQ",
    rls_revision=2,
    fallback_volume_name="NA",
)
```
