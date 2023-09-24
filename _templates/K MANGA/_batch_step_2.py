from pathlib import Path
import subprocess as sp
import os

current_dir = Path(__file__).absolute().parent
os.chdir(current_dir)

source_dir = current_dir / "out"
spd_txt_file = current_dir / "_spreads_data.txt"
backup_dir = current_dir / "backup"

spreads_mappings: dict[str, list[str]] = {}
for spread in spd_txt_file.read_text().splitlines():
    if not (spread := spread.strip()):
        continue
    if spread.startswith("#"):
        continue
    ch_n, spd = spread.split(":", 1)
    ch_nfmt = f"c{int(ch_n):03d}"
    cdd = spreads_mappings.get(ch_nfmt, [])
    cdd.append(spd)
    spreads_mappings[ch_nfmt] = cdd

for directory in source_dir.iterdir():
    if not directory.name.startswith("c"):
        continue
    spread_data = spreads_mappings.get(directory.name, [])
    if not spread_data:
        print("Skipping", directory)
        continue
    # Run girls run
    cmd_base = [
        "nmanga",
        "spreads",
        "join",
        "-r",
    ]
    for spd in spread_data:
        cmd_base.extend(["-s", spd])
    cmd_base.append(str(directory))
    print(" ".join(cmd_base))
    sp.run(cmd_base, shell=True)
    dir_backup = directory / "backup"
    if not dir_backup.exists():
        continue
    for file in dir_backup.iterdir():
        file.rename(backup_dir / file.name)
    dir_backup.rmdir()
