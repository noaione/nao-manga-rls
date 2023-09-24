from pathlib import Path

current = Path(__file__).absolute().parent

source_dir = current / "source"

invalids = ["_temp", "_pgc", "out", "prefinal", "source", "backup"]
for source in source_dir.iterdir():
    if not source.is_dir():
        continue

    for cur_dir in current.iterdir():
        if not cur_dir.is_dir():
            continue

        stem = cur_dir.stem
        if stem.startswith("__"):
            continue

        if stem == "pp":
            (current / "pp" / source.name).mkdir(exist_ok=True)
            (current / "pp" / (source.name + "_pgc")).mkdir(exist_ok=True)
        else:
            (current / stem / source.name).mkdir(exist_ok=True)
