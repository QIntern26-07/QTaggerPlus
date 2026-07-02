# Setup

This project uses [uv](https://docs.astral.sh/uv/) for Python environment and dependency management. Python 3.12+ is required.

## 1. Install uv

macOS / Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

## 2. Create the virtual environment

```bash
uv venv
```

This creates a `.venv` directory using the Python version pinned in `.python-version` (3.12).

## 3. Sync dependencies

```bash
uv sync
```

This installs all dependencies from `pyproject.toml`/`uv.lock` into `.venv`.

## 4. Activate the environment

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\activate
```

## 5. Run the project

```bash
uv run main.py
```

`uv run` automatically uses the synced environment without needing manual activation.
