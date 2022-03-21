#!/media/sdac/mizore/py36env/bin/python
"""
Splits volume into chapters with the page mapping
to properly split it into proper chapters.

Make sure the filename is only number page.

Please follow the current mapping format like this:
- [[FIRST_PAGE, LAST_PAGE], CHAPTER_NUMBER]

This version use regex to capture the page number.
The default regex follow this format:
- Manga Title - (v01) - p178 [dig] [Group] [Uploader].jpg
- Manga Title - v01 - p178 [dig] [Group] [Uploader].jpg

You can adjust it, make sure it catch the page number.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Union
from zipfile import ZipFile, ZipInfo

try:
    _, vol_num, zip_file_path, output_loc = sys.argv
    if isinstance(output_loc, list):
        output_loc = output_loc[0]
    if not vol_num.isdigit():
        print("<volume_num> is not a number!")
    vol_num = int(vol_num)
except ValueError:
    print(f"Usage: {sys.argv[0]} <volume_num> <zipfile> <output_loc>")
    sys.exit(1)


base_re = re.compile(r"Manga Title - \(?v[\d]{1,2}\)? - p(?:([\d]{1,4})(?:-)?([\d]{1,4})?).*")
current_mapping = [
    [[0, 19], 1],
    [[20, 47], 2],
    [[48, 73], 3],
    [[74, 97], 4],
    [[98, 135], 5],
    [[136], 6]
]
current_volume = vol_num


def number(filename: str) -> List[int]:
    # Remove until pXXX
    filename = re.sub(base_re, r"\1-\2", filename)
    if filename.endswith("-"):
        filename = filename[:-1]
    name, _ = os.path.splitext(filename)
    try:
        first, second = name.split("-")
        return [int(first), int(second)]
    except ValueError:
        return [int(name)]


def be_number(number: Union[int, float]) -> str:
    if isinstance(number, int):
        return f"{number:03d}"
    base, floating = str(number).split(".")
    return f"{int(base):03d}.{floating}"


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


target_dir = Path(__file__).absolute().parent / output_loc
print(f"[*] Reading: {zip_file_path}")
zip_data = ZipFile(zip_file_path)
zip_contents = zip_data.filelist
zip_contents.sort(key=lambda x: x.filename)

collected_chapters: Dict[str, List[ZipInfo]] = {}
for chapter_direction in current_mapping:
    page_excerpt = chapter_direction[0]
    chapter_num = be_number(chapter_direction[1])
    final_filename = f"{current_volume:02d}.{chapter_num}"

    range_excerpt = None
    if len(page_excerpt) > 1:
        range_excerpt = list(range(page_excerpt[0], page_excerpt[1] + 1))
    collected_pages: List[ZipInfo] = []
    for zip_info in zip_contents:
        page_number = number(zip_info.filename)
        if len(page_number) == 2:
            page_number = page_number[1]
        else:
            page_number = page_number[0]

        if range_excerpt:
            if page_number in range_excerpt:
                collected_pages.append(zip_info)
        else:
            if page_number >= page_excerpt[0]:
                collected_pages.append(zip_info)
    print(f"[*] {final_filename} => {len(collected_pages)} pages")
    collected_chapters[final_filename] = collected_pages

print(f"[*] Saving to each chapter to {target_dir}")
for chapter_info, chapters_file in collected_chapters.items():
    chapter_secure_target = secure_filename(chapter_info)

    zip_target = target_dir / f"{chapter_info}.cbz"
    zip_target.parent.mkdir(parents=True, exist_ok=True)
    if zip_target.exists():
        print(f"[?][!] Skipping: {chapter_info}")
        continue

    print(f"[?][+] Writing: {chapter_info}")
    zip_target_file = ZipFile(zip_target, "w")
    for chapter in chapters_file:
        zip_target_file.writestr(chapter, zip_data.read(chapter))
    zip_target_file.close()
zip_data.close()
print()
