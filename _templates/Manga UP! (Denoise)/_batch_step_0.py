from pathlib import Path
import subprocess as sp
import os

current_dir = Path(__file__).absolute().parent
os.chdir(current_dir)

source_dir = current_dir / "demangle"
target_dir = current_dir / "out"

for directory in source_dir.iterdir():
    if not directory.name.startswith("c"):
        continue
    (target_dir / directory.stem).mkdir(parents=True, exist_ok=True)
    # Run girls run
    cmd_base = [
        "magick",
        "mogrify",
        "-format",
        "png",
        "-alpha",
        "off",
        "-monitor",
        "-path",
        f".\\out\\{directory.stem}",
        str(directory / "*.tif")
    ]
    print(" ".join(cmd_base))
    sp.run(cmd_base, shell=True)
