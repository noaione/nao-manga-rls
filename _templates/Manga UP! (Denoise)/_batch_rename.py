from pathlib import Path
import os

current_dir = Path(__file__).absolute().parent
os.chdir(current_dir)

credit_name = "nao"
name_fmt = "I'm Not Even an NPC In This Otome Game! - {fn} (NA) - {pg} [web] [{tt}] [Square Enix] [{cc}]"  # noqa

source_dir = current_dir / "denoised"
title_data = {}
for title in (current_dir / "_chapters_name.txt").read_text().splitlines():
    if not (title := title.strip()):
        continue
    if title.startswith("#"):
        continue
    ch_num, ch_title = title.split(": ", 1)
    title_data[str(int(ch_num))] = ch_title

for directory in source_dir.iterdir():
    if not directory.name.startswith("c"):
        continue
    dir_num = int(directory.name[1:])
    all_contents = list(
        filter(lambda x: x.name.endswith(".png"), directory.iterdir())
    )
    for to_rename in all_contents:
        pg_num = to_rename.stem
        # First to last
        ch_title = title_data.get(str(dir_num))
        tname = name_fmt.format(
            fn=directory.stem, pg=pg_num, tt=ch_title, cc=credit_name
        )
        to_rename.rename(to_rename.parent / f"{tname}.png")
