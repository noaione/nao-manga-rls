# [nao] Manga Release Scripts

This repo contains stuff that I use to release and collect manga from a certain cat website.

## Requirements
- Python 3.6+
- imagemagick (for [`multi-merge-spreads.py`](multi-merge-spreads.py))

## Scripts
### `auto-split` and `auto-split-no-title`
This script should help you to split automatically via regex a manga volume into chapters.

The first one should be used if the filename contains the chapter title, while the second one used if there is no chapter title in the filename.

You need to modify the regex to match the filename.

### `manual-split` and `manual-split-no-title`
This script works the same as `auto-split` but the differences that you need to manually map the pages into chapters.

This should be used if the filename is only a page number.

You need to modify mainly this part:
```py
[[0, 19], 1],
[[20, 47], 2],
[[48, 73], 3],
[[74, 97], 4],
[[98, 135], 5],
[[136], 6]
```

The first list contains another list that will match the start-end of the pages, if there's no second number defined it will assume match everything starting from the first number.

The second part of the list is the chapter number.

In the `manual-split` version, you can set the key name into the chapter title.

### `manual-split-regex` and `manual-split-regex-no-title`
Same as `manual-split` this version utilize regex to match the filename and capture the page number.

### `multi-merge-spreads`
Used to merge splitted page into a proper spreads, you need to modify this part to make the proper spreads:
```py
spreads_mappings = [
    [3, 4],
    [31, 32],
    [3, 4, 31],
]
```

The list contains a list that contains the page that should be merged.<br />
In the example, we merge this page:
- Page 3 and 4
- Page 31 and 32
- Page 3, Page 4, and Page 31

### `prepare-release` and `prepare-release-no-title`
Both of this script is a renamer for my release, my release use this following format on the filename:
- Manga Title - c001 (v01) - p000 [Cover] [dig] [Chapter 1] [Publisher] [nao] {HQ}
- Manga Title - c001 (v01) - p000 [dig] [Chapter 1] [Publisher] [nao] {HQ}

If you use the no title version, the `[Chapter 1]` part will be removed.

You need to modify `current_mapping`, `special_naming` and `target_fmt`.

Current mapping follows the `manual-split` formatting, while special naming is for adding that extra thing before `[dig]`.

The key is the page number, the value is the extra.

And target_fmt is the filename format, you should only change `Manga Title` and `Publisher`.
