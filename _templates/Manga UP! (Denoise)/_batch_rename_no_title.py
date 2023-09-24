from pathlib import Path
import os

current_dir = Path(__file__).absolute().parent
os.chdir(current_dir)

# change both of this if needed
credit_name = "nao"
name_fmt = "I'm Not Even an NPC In This Otome Game! - {fn} (NA) - {pg} [web] [Square Enix] [{cc}]"  # noqa

source_dir = current_dir / "denoised"

for directory in source_dir.iterdir():
    if not directory.name.startswith("c"):
        continue
    dir_num = int(directory.name[1:])
    all_contents = list(
        filter(lambda x: x.name.endswith(".png"), directory.iterdir())
    )
    for to_rename in all_contents:
        pg_num = to_rename.stem
        tname = name_fmt.format(
            fn=directory.stem, pg=pg_num, cc=credit_name
        )
        to_rename.rename(to_rename.parent / f"{tname}.png")
