# Changelog

- [v0.1.0](#010)
  - [v0.1.1](#011)
- [v0.2.0](#020)
  - [v0.2.1](#021)
  - [v0.2.2](#022)
  - [v0.2.3](#023)
  - [v0.2.4](#024)
- [v0.3.0](#030)
  - [v0.3.1](#031)
  - [v0.3.2](#032)
  - [v0.3.3](#033)
  - [v0.3.4](#034)
- [v0.4.0](#040)
  - [v0.4.1](#041)
  - [v0.4.2](#042)
  - [v0.4.3](#043)
  - [v0.4.4](#044)
- [v0.5.0](#050)
- [v0.6.0](#060)
- [v0.7.0](#070)
  - [v0.7.1](#071)
  - [v0.7.2](#072)
- [v0.8.0](#080)
  - [v0.8.1](#081)
  - [v0.8.2](#082)
- [v0.9.0](#090)
  - [v0.9.1](#091)

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
