from pathlib import Path
import subprocess
import sys


API_ROOT = Path(__file__).resolve().parents[1]


def run_local_script(relative_script_path: str, timeout_seconds: int = 30) -> dict:
    script_path = API_ROOT / relative_script_path

    if not script_path.exists():
        return {
            "ok": False,
            "reason": "missing_script",
            "script": str(script_path),
        }

    command = [sys.executable, str(script_path)]

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(API_ROOT),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "reason": "timeout",
            "script": str(script_path),
        }

    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "script": str(script_path),
    }