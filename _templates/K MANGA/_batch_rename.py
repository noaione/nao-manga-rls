from pathlib import Path
import os

current_dir = Path(__file__).absolute().parent
os.chdir(current_dir)

credit_name = "nao"
name_fmt = "The Classroom of a Black Cat and a Witch - {fn} (NA) - p{pg:03d} [web] [{tt}] [Kodansha Comics] [{cc}]"  # noqa

source_dir = current_dir / "out4"
title_data = {}
for title in (current_dir / "_chapters_name.txt").read_text().splitlines():
    if not (title := title.strip()):
        continue
    if title.startswith("#"):
        continue
    try:
        ch_num, ch_title = title.split(": ", 1)
    except ValueError:
        continue
    title_data[str(int(ch_num))] = ch_title

for directory in source_dir.iterdir():
    if not directory.name.startswith("c"):
        continue
    dir_num = int(directory.name[1:])
    all_contents = list(
        filter(lambda x: x.name.endswith(".png"), directory.iterdir())
    )
    for to_rename in all_contents:
        pg_num = int(to_rename.stem.split("_", 1)[0]) - 1
        # First to last
        ch_title = title_data.get(str(dir_num))
        tname = name_fmt.format(
            fn=directory.stem, pg=pg_num, tt=ch_title, cc=credit_name
        )
        to_rename.rename(to_rename.parent / f"{tname}.png")
