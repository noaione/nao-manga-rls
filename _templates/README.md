# Release Templates

The following folder contains all the extra additional scripts that I use to make my releases.

Just copy the templates and create the folder for that specific manga.

## Templates
- [K MANGA](https://github.com/noaione/nao-manga-rls/tree/master/_templates/K%20MANGA) — Grayscaling, rename, spreads join
- [Manga UP!](https://github.com/noaione/nao-manga-rls/tree/master/_templates/Manga%20UP!) — AVIF to PNG, rename, 4-bit conversion, spreads join
- [Manga UP! w/ Denoise](https://github.com/noaione/nao-manga-rls/tree/master/_templates/Manga%20UP!%20(Denoise)) — AVIF to PNG, denoise BM3D + SMDegrain, rename, 4-bit conversion, spreads join
- [Square Enix Manga](https://github.com/noaione/nao-manga-rls/tree/master/_templates/Square%20Enix) — Fix blurry mess for Square Enix volume releases, combined with chaiNNer
- [Upscaling](https://github.com/noaione/nao-manga-rls/tree/master/_templates/Upscale) — Upscale low quality image (usually Yen Press stuff), combined with chaiNNer

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