#!/media/sdac/mizore/py36env/bin/python
import re
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
        return f"{int(base):03d}x{int(floating) - 4}"


img_regex = re.compile(r".*\- (?P<vol>v[\d]{1,2}) - p(?P<a>[\d]{1,3})\-?(?P<b>[\d]{1,3})?")
valid_images = [".jpeg", ".jpg", ".png", ".gif", ".jiff", ".webp"]

all_images: List[Path] = []
for file in target_dir.rglob("*"):
    if file.is_file() and file.suffix.lower() in valid_images:
        all_images.append(file)

current_mapping = {
    "1": [[0, 41], 1],
    "2": [[42, 67], 2],
    "3": [[68, 85], 3],
    "4": [[86, 99], 4],
    "5": [[100, 113], 5],
    "6": [[114, 129], 6],
    "7": [[130, 145], 7],
    "8": [[146], 8],
}
special_naming: Dict[int, str] = {
    0: "Cover",
    # 1: "Inside Cover",
    # 3: "ToC",
    # 159: "Afterword",
}
target_fmt = "Chronicles of an Aristocrat Reborn in Another World - c{ch} ({vol}) - p{pg}{ex}[dig] [Seven Seas] [nao] {HQ}"  # noqa


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
    for name, ch_num in special_naming.items():
        if ch_num == p01_copy:
            extra_name = f" [{name}] "

    chapter_num = selected_range.bnum
    final_name = target_fmt.format(
        ch=chapter_num, vol=vol, ex=extra_name, pg=p01, HQ=r"{HQ}"
    ) + extension

    new_name = target_dir / final_name
    image.rename(new_name)
