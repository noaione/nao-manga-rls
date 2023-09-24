# Release Templates

The following folder contains all the extra additional scripts that I use to make my releases.

Just copy the templates and create the folder for that specific manga.

## Templates
- [K MANGA]() — Grayscaling, rename, spreads join
- [Manga UP!]() — AVIF to PNG, rename, 4-bit conversion, spreads join
- [Manga UP! w/ Denoise]() — AVIF to PNG, denoise BM3D + SMDegrain, rename, 4-bit conversion, spreads join
- [Upscaling]() — Combined with chaiNNer

## Common Folders
- `source` — Where to put the source image, named with `c` prefix or `v` prefix for chapter and volume, followed by their chapter number (3 padded) or volume number (2 padded)
- `out` — The output of first conversion (for anything other than Upscaling)
- `out4` — The output of 4-bit conversion
- `denoised` — The denoised image
- `demangle` — Legacy, used to fix broken AVIF conversion for newer imagemagick
- `_toned` — Source full-tone color or grayscale image
- `_pgc` — Cleaned/upscaled image from `_toned`
- `_temp` — Cleaned/upscaled image from `source`
- `pp` — Post-processed image, usually before being used as the final image