from pathlib import Path
import os
import subprocess
import sys


API_ROOT = Path(__file__).resolve().parents[1]


def run_local_script(script_path_input: str, timeout_seconds: int = 30) -> dict:
    raw_path = Path(script_path_input)
    script_path = raw_path if raw_path.is_absolute() else (API_ROOT / raw_path).resolve()

    if not script_path.exists():
        return {
            "ok": False,
            "reason": "missing_script",
            "script": str(script_path),
        }

    command = [sys.executable, str(script_path)]
    env = os.environ.copy()
    is_serverless = any(
        env.get(name)
        for name in ("VERCEL", "AWS_EXECUTION_ENV", "AWS_LAMBDA_FUNCTION_NAME", "LAMBDA_TASK_ROOT")
    )
    if str(API_ROOT).startswith("/var/task"):
        is_serverless = True
    if is_serverless and not env.get("RUNTIME_DATA_DIR"):
        env["RUNTIME_DATA_DIR"] = "/tmp/coffee-market-data"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(API_ROOT),
            env=env,
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