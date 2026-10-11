"""Reserve separate output folders for desktop and Explorer runs."""

from pathlib import Path


def create_run_folder(source: Path, output_root: Path, strings: int, profile: str) -> Path:
    """Atomically reserve a folder, keeping earlier runs and concurrent runs intact."""
    output_root.mkdir(parents=True, exist_ok=True)
    source_name = (source.name or "Music") if source.is_dir() else source.stem
    name = f"{source_name}.{strings}-string.{profile}"
    number = 1
    while True:
        folder = output_root / (name if number == 1 else f"{name}.{number}")
        try:
            folder.mkdir()
        except FileExistsError:
            number += 1
        else:
            return folder
