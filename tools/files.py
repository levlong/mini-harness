from pathlib import Path

from pydantic import BaseModel, Field

MAX_READ_CHARS = 20_000


class PathArgs(BaseModel):
    path: str = Field(description="Path relative to the workspace")


class WriteArgs(BaseModel):
    path: str = Field(description="Path relative to the workspace")
    content: str = Field(description="Full file content to write")


def _safe(workspace: Path, rel: str) -> Path:
    """Resolve a path and refuse anything outside the workspace."""
    p = (workspace / rel).resolve()
    if not p.is_relative_to(workspace):
        raise PermissionError(f"'{rel}' is outside the workspace")
    return p


def _existing(workspace: Path, rel: str) -> Path:
    """Like _safe, but with a short error the model can act on if the path is missing."""
    p = _safe(workspace, rel)
    if not p.exists():
        raise FileNotFoundError(f"'{rel}' does not exist. Use list_dir to see what is there")
    return p


def register(reg, workspace: Path):
    @reg.register("list_dir", "List files and folders in a directory", PathArgs)
    def list_dir(path: str) -> str:
        p = _existing(workspace, path)
        items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
        return "\n".join(f"{i.name}/" if i.is_dir() else i.name for i in items) or "(empty)"

    @reg.register("read_file", "Read a UTF-8 text file", PathArgs)
    def read_file(path: str) -> str:
        text = _existing(workspace, path).read_text(encoding="utf-8")
        if len(text) > MAX_READ_CHARS:  # protect the context window
            return text[:MAX_READ_CHARS] + f"\n...[truncated, {len(text)} chars total]"
        return text

    @reg.register("write_file", "Create or overwrite a text file", WriteArgs)
    def write_file(path: str, content: str) -> str:
        p = _safe(workspace, path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"
