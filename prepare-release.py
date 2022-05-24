#!/media/sdac/mizore/py36env/bin/python
import re
import subprocess as sp
import sys
from pathlib import Path
from typing import Dict, List

try:
    target_dir = Path(sys.argv[1])
except IndexError:
    print(f"Usage: {sys.argv[0]} <target_dir>")
    exit(1)


class SimpleRange:
    def __init__(self, number: int, name: str, range: List[int], is_single: bool = False):
        self.number = number
        self.name = name
        self.range = range
        self.is_single = is_single

    def __repr__(self):
        if isinstance(self.number, float):
            return f"<SimpleRange c{self.number} - {self.name}>"
        return f"<SimpleRange c{self.number:03d} - {self.name}>"

    @property
    def bnum(self):
        if isinstance(self.number, int):
            return f"{self.number:03d}"
        base, floating = str(self.number).split(".")
        floating = int(floating)
        if floating - 4 >= 1:
            # Handle split chapter (.1, .2, etc)
            floating -= 4
        return f"{int(base):03d}x{floating}"


def test_exiftool():
    # Test exiftool
    try:
        sp.run(["exiftool", "-ver"], check=True)
    except sp.CalledProcessError:
        print("Unable to find exiftool")
        exit(1)


def inject_metadata(im_path: Path):
    test_exiftool()
    base_cmd = ["exiftool"]
    update_tags = {
        "XPComment": "noaione@protonmail.com",
        "Artist": "noaione@protonmail.com",
        "XPAuthor": "noaione@protonmail.com",
    }
    for tag, value in update_tags.items():
        base_cmd.append(f'-{tag}="{value}"')
    base_cmd.append(str(im_path))
    proc = sp.Popen(base_cmd, stdout=sp.PIPE, stderr=sp.PIPE)
    proc.wait()


test_exiftool()
img_regex = re.compile(r".*\- (?P<vol>v[\d]{1,2}) - p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?")
valid_images = [".jpeg", ".jpg", ".png", ".gif", ".jiff", ".webp"]

all_images: List[Path] = []
for file in target_dir.glob("*"):
    if file.is_file() and file.suffix.lower() in valid_images:
        all_images.append(file)

current_mapping = {
    "That Jerk Won't Fall for Me": [[0, 41], 1],
    "P.E. with That Jerk": [[42, 67], 2],
    "Sketching with That Jerk": [[68, 85], 3],
    "That Fiendish Jerk": [[86, 99], 4],
    "In the Library with That Jerk": [[100, 113], 5],
    "Love and That Jerk": [[114, 129], 6],
    "That Girl's Wallpaper": [[130, 145], 7],
    "Cosplaying with That Jerk": [[146], 8],
}
special_naming: Dict[int, str] = {
    0: "Cover",
    # 1: "Inside Cover",
    # 3: "ToC",
    # 159: "Afterword",
}
target_fmt = "Manga Title - c{ch} ({vol}) - p{pg}{ex}[dig] [{t}] [Publisher] [nao] {HQ}"  # noqa


generate_ranges: List[SimpleRange] = []
for chapter_name, chapter_ranges in current_mapping.items():
    page_excerpt, ch_num = chapter_ranges
    range_excerpt = None
    if len(page_excerpt) > 1:
        range_excerpt = list(range(page_excerpt[0], page_excerpt[1] + 1))

    if range_excerpt is None:
        generate_ranges.append(
            SimpleRange(
                ch_num,
                chapter_name,
                page_excerpt,
                True,
            )
        )
    else:
        generate_ranges.append(SimpleRange(ch_num, chapter_name, range_excerpt))

for image in all_images:
    extension = image.suffix.lower()
    title_match = re.match(img_regex, image.name)

    p01 = title_match.group("a")
    p01_copy = int(title_match.group("a"))
    p02 = title_match.group("b")
    vol = title_match.group("vol")
    if p02 is not None:
        p01 = f"{p01}-{p02}"

    # print(p01_copy)
    selected_range: SimpleRange = None
    for generated in generate_ranges:
        if generated.is_single:
            if p01_copy >= generated.range[0]:
                selected_range = generated
                break
        else:
            if p01_copy in generated.range:
                selected_range = generated
                break
    extra_name = " "
    for ch_num, name in special_naming.items():
        if ch_num == p01_copy:
            extra_name = f" [{name}] "

    title_name = selected_range.name
    chapter_num = selected_range.bnum
    final_name = target_fmt.format(
        ch=chapter_num, vol=vol, ex=extra_name, pg=p01, t=title_name, HQ=r"{HQ}"
    ) + extension

    new_name = target_dir / final_name
    # Before rename, inject metadata
    try:
        inject_metadata(image)
    except Exception:
        pass
    image.rename(new_name)
