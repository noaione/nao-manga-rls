#!/media/sdac/mizore/py36env/bin/python
"""
This script will help merge spreads together.
You need imagemagick installed in PATH.

You also need to change spreads_mapping on line 50.
The format is just the page number to be merged.

This use regex to match all images in the folder.
The regex should match something like this:
- Manga Title - v01 - p178.jpg

You can adjust it, make sure it catch the page number.
The first part should have "a" as it's group name
and the last part should have "b" as it's group name and optional.
"""

import re
import sys
import subprocess as sp
import os
from shutil import move as mv
from pathlib import Path
from typing import Dict, List, Match

try:
    target_dir = Path(sys.argv[1]).absolute()
except IndexError:
    print(f"Usage: {sys.argv[0]} <target_dir>")
    exit(1)

backup_dir = target_dir / "backup"
backup_dir.mkdir(exist_ok=True)
img_regex = re.compile(r"(?P<t>.*)\- (?P<vol>v[\d]{1,2}) - p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?")
valid_images = [".jpeg", ".jpg", ".png", ".gif", ".jiff", ".webp"]

all_images: List[Path] = []
for file in target_dir.glob("*"):
    if file.is_file() and file.suffix.lower() in valid_images:
        all_images.append(file)
all_images.sort(key=lambda x: x.name)
compiled_results: Dict[Path, Match[str]] = {}
for image in all_images:
    title_match = re.match(img_regex, os.path.basename(image.name))
    if title_match is None:
        print(f"Unmatched: {image.name}")
        exit(1)
    compiled_results[image] = title_match

spreads_mappings = [
    [3, 4],
    [31, 32],
    [3, 4, 31],
]

images_collections: List[Dict[Path, Match[str]]] = []
for spread in spreads_mappings:
    if len(spread) < 1:
        continue
    image_collect: Dict[str, Match[str]] = {}
    for image, image_re in compiled_results.items():
        b_part = image_re.group("b")
        if b_part:
            continue
        a_part = int(image_re.group("a"))
        print(a_part, spread)
        if a_part in spread:
            image_collect[image] = image_re
    images_collections.append(image_collect)


for spreads_join in images_collections:
    magick_args = ["magick", "convert"]
    image_sets: List[str] = []
    for image, image_re in spreads_join.items():
        image_sets.append(str(image))
    image_sets.reverse()
    magick_args.extend(image_sets)

    extensions = []
    for image in spreads_join.keys():
        extensions.append(image.suffix)
    select_ext = ".jpg"
    if ".png" in extensions:
        select_ext = ".png"

    first_val = list(spreads_join.values())[0]
    last_val = list(spreads_join.values())[-1]
    title = first_val.group("t").strip()
    final_filename = f"{title} - {first_val.group('vol')} - p{first_val.group('a')}-{last_val.group('a')}"
    final_filename += select_ext
    magick_args.extend(["+append", str(target_dir / final_filename)])
    print(f"Merging to: {final_filename}")
    sp.check_call(magick_args)

for spreads_join in images_collections:
    for image in spreads_join.keys():
        mv(image, backup_dir / os.path.basename(image.name))
