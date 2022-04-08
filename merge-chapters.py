#!/media/sdac/mizore/py36env/bin/python
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import List
from zipfile import ZipFile

another_arguments = sys.argv[1:]

collected_cbz_files: List[Path] = []
for argument in another_arguments:
    if not argument.endswith(".cbz"):
        print(f"{argument} is not a cbz file")
        exit(1)

    set_path = Path(argument)
    if not set_path.exists():
        print(f"{set_path} does not exist")
        exit(1)

    collected_cbz_files.append(Path(argument))


target_file = collected_cbz_files[0]
target_io = BytesIO()
target_zip_data = ZipFile(target_io, "w")

for cbz_file in collected_cbz_files:
    print(f"[*] Merging: {cbz_file}")
    with ZipFile(cbz_file, "r") as zip_data:
        for files in zip_data.filelist:
            target_zip_data.writestr(os.path.basename(files.filename), zip_data.read(files))
    os.remove(cbz_file)
target_zip_data.close()

with target_file.open("wb") as target_zip_file:
    print(f"[*] Writing: {target_file}")
    target_zip_file.write(target_io.getvalue())
