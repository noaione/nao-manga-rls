from pathlib import Path
import os

current_dir = Path(__file__).absolute().parent
os.chdir(current_dir)

# change both of this if needed
credit_name = "nao"
name_fmt = "The Classroom of a Black Cat and a Witch - {fn} (NA) - p{pg:03d} [web] [Kodansha Comics] [{cc}]"  # noqa

source_dir = current_dir / "out4"

for directory in source_dir.iterdir():
    if not directory.name.startswith("c"):
        continue
    dir_num = int(directory.name[1:])
    all_contents = list(
        filter(lambda x: x.name.endswith(".png"), directory.iterdir())
    )
    for to_rename in all_contents:
        pg_num = int(to_rename.stem.split("_", 1)[0]) - 1
        tname = name_fmt.format(
            fn=directory.stem, pg=pg_num, cc=credit_name
        )
        to_rename.rename(to_rename.parent / f"{tname}.png")
