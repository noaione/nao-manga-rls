# [nao] Manga Release Scripts

This repo contains stuff that I use to release and collect manga from a certain cat website.

All my release use the `[nao]` tag.

## Released Stuff
See here: https://shigoto.n4o.xyz/manga

**Dropped**
- BAKEMONOGATARI (v13)
- Chainsaw Man (v09)
- Medaka Kuroiwa is Impervious to My Charms (v01)
- My Deer Friend Nekotan (v01)
- The Ice-Guy and His Cool Female Colleague (Taken over by someone else)

Anything in Dropped means I already ripped it once, but I dropped it.

You can find it on a certain cat website.

## Requirements
- Python 3.7+
- imagemagick (for `level` and `spreads` command)
- exiftool (for `releases`, optional)

## Scripts
This repo also have a module or script to release and split stuff up.

To install, you need git since I'm not publishing this to PyPi.
After that you can run this:

```sh
pip install -U git+https://github.com/noaione/nao-manga-rls.git@module-rewrite#egg=nmanga
```

This will install this project which then you can use the command `nmanga` to execute everything.

### Commands

#### `autosplit`
Same as the old auto split script, this command accept some parameter.

```py
Usage: nmanga autosplit [OPTIONS] FOLDER_OR_ARCHIVE_FILE

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
- `Manga Title - c001 (v01) - p001 [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg`
- `Manga Title - c001 (v01) - p000 [Cover] [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg`
- `Manga Title - c001 (v01) - p000 [Cover] [dig] [Publisher Name?] [Group] {HQ}.jpg`
- `Manga Title - c001 (v01) - p001 [dig] [Publisher Name?] [Group] {HQ}.jpg`

This should match most of the release from a certain cat website (`danke`, `LuCaZ`, `1r0n`, etc.)

#### `level`
Automatically batch color level an archive or folder containing images.<br />
This can be useful to some VIZ release since sometimes the "black" color is not really that "black" (ex. `#202020` instead of `#000000`)

Recommended parameters:
- Darkest color at `#231f20`: `-l 13 -h 100 --gray -skip`
- Darkest color at `#202020`: `-l 13 -h 100 --gray -skip`
- Darkest color page at `#000`: `-l 0 -h 100 --gray` (Set as grayscale)

### `manualsplit`
Same as the old manual split script, this command accept some parameter and will then ask a series of question.

```py
Usage: nmanga manualsplit [OPTIONS] FOLDER_OR_ARCHIVE_FILE

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

#### `releases`
Create a release from `comix` formatted filename.

```py
Usage: nmanga releases [OPTIONS] FOLDER_OR_ARCHIVE_FILE

  Prepare a release of a manga series.

Options:
  -t, --title TEXT           The title of the series  [required]
  -y, --year INTEGER         The year of the series release
  -pub, --publisher TEXT     The publisher of the series  [required]
  -c, --credit TEXT          The ripper credit for this series  [default: nao]
  -e, --email TEXT           The ripper email for this series  [default:
                             noaione@protonmail.com]
  -hq, --is-high-quality     Whether this is a high quality release
  --tag / --no-tag           Do exif metadata tagging on the files.  [default:
                             tag]
  -ee, --exiftool-exec TEXT  Path to the exiftool executable  [default:
                             exiftool]
  -h, --help                 Show this message and exit.
```

`--title`, the series title<br />
`--year`, the series year (will be used for exif tagging)<br />
`--publisher`, the publisher<br />
`--credit`, the ripped/group name<br />
`--email`, will be used for exif tagging<br />
`--is-high-quality`, mark the release as HQ (add `{HQ}` to filename)<br />
`--tag/--no-tag`, do exif tagging.

After you mark the folder or archive, you will be asked a series of question.
You can enter all the information and after that the program will rename everything and tag according to your specification.

This use the same inquiring method as `manualsplit`

The filename also must match something like this:
- `Manga Title - vXX - pXXX`

#### `spreads`
Join spread pages together into a single page.

```py
Usage: nmanga spreads [OPTIONS] FOLDER_OR_ARCHIVE_FILE

  Join multiple spreads into a single image

Options:
  -q, --quality FLOAT RANGE  The quality of the output image  [default: 100.0;
                             1.0<=x<=100.0]
  -s, --spreads A-B          The spread information, can be repeated and must
                             contain something like: 1-2  [required]
  -r, --reverse              Reverse the order of the spreads (manga mode)
  -me, --magick-exec TEXT    Path to the magick executable  [default: magick]
  -h, --help                 Show this message and exit.
```

`--quality`, the output quality (mainly used as jpg export)<br />
`--spreads`, the spread information, can be repeated (ex: `-s 3-4 -s 99-100` will merge page 3 and 4 together.)<br />
`--reverse`, reverse the order of the spread (recommended for manga/RTL)

The filename also must match something like this:
- `Manga Title - vXX - pXXX`

This will also make a `backup` folder which contains the unmerged images (in case something went wrong.)