"""Per-user Windows Explorer integration; no file associations are replaced."""

from __future__ import annotations

import argparse
import ctypes
import subprocess
import sys
from pathlib import Path

AUDIO_EXTENSIONS = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
MENU_KEY = "BassTranscriber"
COMMANDS_KEY = "BassTranscriber.Commands"
PROFILES = {
    "balanced": "Balanced",
    "avoid_open": "Punish open strings",
    "combined": "Combined",
}
OUTPUT_MODES = {"tonelib_only": "ToneLib Only", "debug": "Debug"}


def _menu_roots() -> list[str]:
    return [
        rf"Software\Classes\SystemFileAssociations\{extension}\shell\{MENU_KEY}"
        for extension in AUDIO_EXTENSIONS
    ] + [rf"Software\Classes\Directory\shell\{MENU_KEY}"]


def menu_entries(pythonw: Path, output_root: Path) -> dict[str, dict[str, str]]:
    """Describe only the registry keys owned by this application."""
    entries: dict[str, dict[str, str]] = {}
    for root in _menu_roots():
        entries[root] = {
            "MUIVerb": "Generate bass tablature",
            "ExtendedSubCommandsKey": COMMANDS_KEY,
            "MultiSelectModel": "Single",
        }
    for mode, mode_label in OUTPUT_MODES.items():
        mode_key = f"{COMMANDS_KEY}.{mode}"
        entries[rf"Software\Classes\{COMMANDS_KEY}\shell\{mode}"] = {
            "MUIVerb": mode_label,
            "ExtendedSubCommandsKey": mode_key,
        }
        for strings, tuning in ((4, "EADG"), (5, "BEADG")):
            group = rf"Software\Classes\{mode_key}\shell\{strings}strings"
            profile_key = f"{mode_key}.{strings}"
            entries[group] = {
                "MUIVerb": f"{strings}-string bass ({tuning})",
                "ExtendedSubCommandsKey": profile_key,
            }
            for profile, label in PROFILES.items():
                verb = rf"Software\Classes\{profile_key}\shell\{profile}"
                entries[verb] = {"MUIVerb": label, "MultiSelectModel": "Single"}
                command = subprocess.list2cmdline(
                    [
                        str(pythonw),
                        "-m",
                        "bass_transcriber.shell_job",
                        "--output-root",
                        str(output_root),
                        "--strings",
                        str(strings),
                        "--fingering-profile",
                        profile,
                        "--mode",
                        mode,
                    ]
                )
                # Explorer substitutes the selected file or folder; preserve spaces.
                entries[verb + r"\command"] = {"": command + ' "%1"'}
    return entries


def remove_menu() -> None:
    """Remove our file/folder menus and current or legacy command groups."""
    import winreg

    def delete_tree(path: str) -> None:
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as key:
                children = [winreg.EnumKey(key, i) for i in range(winreg.QueryInfoKey(key)[0])]
        except FileNotFoundError:
            return
        for child in children:
            delete_tree(path + "\\" + child)
        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, path)

    for root in _menu_roots():
        delete_tree(root)
    delete_tree(rf"Software\Classes\*\shell\{MENU_KEY}")
    for name in (COMMANDS_KEY, f"{COMMANDS_KEY}.4", f"{COMMANDS_KEY}.5"):
        delete_tree(rf"Software\Classes\{name}")
    for mode in OUTPUT_MODES:
        for name in (
            f"{COMMANDS_KEY}.{mode}",
            f"{COMMANDS_KEY}.{mode}.4",
            f"{COMMANDS_KEY}.{mode}.5",
        ):
            delete_tree(rf"Software\Classes\{name}")


def main() -> int:
    """Install or remove the local repository's Explorer menu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("install", "uninstall"))
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Explorer integration is available only on Windows")
    if args.action == "install":
        repository = Path(__file__).resolve().parents[2]
        if not (repository / "pyproject.toml").is_file():
            parser.error("Install the context menu from this repository's virtual environment")
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if not pythonw.is_file():
            parser.error(f"Windowed Python interpreter is missing: {pythonw}")
        import winreg

        remove_menu()
        for path, values in menu_entries(pythonw, repository / "outputs").items():
            with winreg.CreateKey(winreg.HKEY_CURRENT_USER, path) as key:
                for name, value in values.items():
                    winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
        print(f"Installed Explorer menu. Results go to {repository / 'outputs'}")
    else:
        remove_menu()
        print("Removed Explorer menu.")
    # Tell Explorer that shell associations have changed without restarting it.
    ctypes.windll.shell32.SHChangeNotify(0x08000000, 0x1000, None, None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
