#!/media/sdac/mizore/py36env/bin/python
"""
Auto split volume into chapters

This script should only be used if your filename match something akin like this:
- Manga Title - c001 (v01) - p000 [Cover] [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg
- Manga Title - c001 (v01) - p001 [dig] [Chapter Title] [Publisher Name?] [Group] {HQ}.jpg

If the [dig] is different, you can change it in the regex below.

To use, you can chmod this file with execute and just run it as is.
Make sure to put it in the same folder as the files you want to split.
"""

import re
import os
from pathlib import Path
from typing import Dict, List
from zipfile import ZipFile, ZipInfo

current_dir = Path(__file__).absolute().parent
# CHANGETHIS is the manga title or anything before the chapter number
volume_re = re.compile(r"CHANGETHIS v(\d+) .*")
# CHANGETHIS is the manga title or anything before the chapter number
# CHANGEPUBLISHER will be the publisher name, it should be ended with `.*` so it will match everything elses.
chapter_re = re.compile(
    r"CHANGETHIS - c(?P<ch>\d+)(?P<ex>x[\d]{1,2})? \(v[\d]+\) - p[\d]+x?[\d]?\-?[\d]+x?[\d]? "
    r".*\[dig] (?:\[(?P<title>.*)\] )?\[CHANGEPUBLISHER.*"
)
cbz_files = list(current_dir.glob("*.cbz"))


def clean_title(title: str):
    if not title:
        return title
    if title.endswith("]"):
        title = title[:-1]
    return title


def secure_filename(fn: str):
    replacement = {
        "/": "／",
        ":": "：",
        "<": "＜",
        ">": "＞",
        '"': "”",
        "'": "’",
        "\\": "＼",
        "?": "？",
        "*": "⋆",
        "|": "｜",
        "#": "",
    }
    for k, v in replacement.items():
        fn = fn.replace(k, v)
    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])"
    )
    fn = re.sub(EMOJI_PATTERN, "_", fn)
    return fn


valid_cbz_files: Dict[str, Path] = {}

for cbz_file in cbz_files:
    match_re = volume_re.match(cbz_file.name)
    if not match_re:
        continue
    volume_num = match_re.group(1)
    if not volume_num:
        continue
    valid_cbz_files[volume_num] = cbz_file


for volume, file_path in valid_cbz_files.items():
    print(f"[?] Processing: v{volume}")
    zip_data = ZipFile(file_path)
    target_path = current_dir / f"v{volume}"
    zip_contents = zip_data.filelist
    zip_contents.sort(key=lambda x: x.filename)

    collected_chapters: Dict[str, List[ZipInfo]] = {}
    for zip_info in zip_contents:
        if zip_info.is_dir():
            continue
        filename = zip_info.filename
        if filename.endswith(".xml"):
            continue
        match_re = chapter_re.match(os.path.basename(filename))
        if not match_re:
            # Show the file name that is not matched
            print(zip_info)
        chapter_num = match_re.group("ch")
        chapter_title = clean_title(match_re.group("title"))
        chapter_extra = match_re.group("ex")
        if chapter_extra:
            chapter_extra = int(chapter_extra[1:])
            chapter_data = f"{chapter_num}.{4 + chapter_extra}"
            if not chapter_title:
                chapter_data += f" - Extra {chapter_extra}"
            else:
                chapter_data += f" - Extra {chapter_extra} - {chapter_title}"
        else:
            chapter_data = chapter_num
            if chapter_title:
                chapter_data += f" - {chapter_title}"
        if chapter_data not in collected_chapters:
            collected_chapters[chapter_data] = []
        collected_chapters[chapter_data].append(zip_info)

    for chapter_info, chapters_file in collected_chapters.items():
        chapter_secure_target = secure_filename(chapter_info)
        zip_target = target_path / f"{volume}.{chapter_secure_target}.cbz"
        zip_target.parent.mkdir(parents=True, exist_ok=True)
        if zip_target.exists():
            print(f"[?][!] Skipping: {zip_target.name}")
            continue
        print(f"[?][+] Writing: {chapter_info}")
        zip_target_file = ZipFile(zip_target, "w")
        for chapter in chapters_file:
            chapter_fn = os.path.basename(chapter.filename)
            zip_target_file.writestr(chapter_fn, zip_data.read(chapter))
        zip_target_file.close()
    zip_data.close()
    print()
