# Bass Transcriber

A local Python project for converting songs into synchronized bass tablature that can be opened in ToneLib Jam.

The refined design and V1 scope are documented in [v1plan.md](v1plan.md).

## Development setup

Install [uv](https://docs.astral.sh/uv/), then run:

```powershell
uv sync
uv run pytest
uv run ruff check .
uv run mypy src
```

The repository pins its uv-managed Python version in `.python-version`.

## CLI

```powershell
uv run bass-transcriber --version
```

