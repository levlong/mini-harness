import shlex
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

# Only these programs may run. Extend deliberately.
ALLOWED = {"ls", "cat", "echo", "pwd", "python", "python3", "pytest", "git", "grep", "wc"}
MAX_OUTPUT_CHARS = 10_000


class CommandArgs(BaseModel):
    command: str = Field(description="A single command, e.g. 'pytest -q'. No pipes or &&.")


def register(reg, workspace: Path):
    @reg.register(
        "run_command",
        f"Run a whitelisted command in the workspace. Allowed: {', '.join(sorted(ALLOWED))}",
        CommandArgs,
    )
    def run_command(command: str) -> str:
        argv = shlex.split(command)
        if not argv or argv[0] not in ALLOWED:
            return f"ERROR: '{argv[0] if argv else ''}' is not allowed"
        # no shell=True: pipes, &&, $(...) are never interpreted
        r = subprocess.run(argv, cwd=workspace, capture_output=True, text=True, timeout=60)
        out = (r.stdout + r.stderr)[:MAX_OUTPUT_CHARS]
        return f"exit={r.returncode}\n{out}"
