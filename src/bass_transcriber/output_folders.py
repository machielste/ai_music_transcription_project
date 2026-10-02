"""Reserve separate output folders for desktop and Explorer runs."""

from pathlib import Path


def create_run_folder(source: Path, output_root: Path, strings: int, profile: str) -> Path:
    """Atomically reserve a folder, keeping earlier runs and concurrent runs intact."""
    output_root.mkdir(parents=True, exist_ok=True)
    name = f"{source.stem}.{strings}-string.{profile}"
    number = 1
    while True:
        folder = output_root / (name if number == 1 else f"{name}.{number}")
        try:
            folder.mkdir()
        except FileExistsError:
            number += 1
        else:
            return folder
