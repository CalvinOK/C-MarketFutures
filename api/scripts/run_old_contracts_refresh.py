from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
API_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[2]


def _python_executable() -> str:
    venv_python = API_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)

    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def main() -> None:
    script = API_ROOT / "data_fetch.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing script: {script}")

    cmd = [_python_executable(), str(script)]
    proc = subprocess.run(
        cmd,
        cwd=str(API_ROOT),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    if proc.stdout.strip():
        print(proc.stdout.strip())
    if proc.returncode != 0:
        if proc.stderr.strip():
            print(proc.stderr.strip())
        raise RuntimeError(f"Contracts refresh failed (exit={proc.returncode})")

    print("[contracts] refresh completed")


if __name__ == "__main__":
    main()