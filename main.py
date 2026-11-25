import sys

from rich.console import Console
from rich.panel import Panel

from agent import run_agent
from config import load_settings
from llm import LLM
from runlog import RunLog
from tools import default_registry

console = Console()


def show(kind: str, data: dict):
    if kind == "tool_call":
        console.print(f"[cyan]step {data['step']}[/] → [bold]{data['name']}[/] {data['args']}")
    elif kind == "tool_result":
        preview = data["result"][:300] + ("…" if len(data["result"]) > 300 else "")
        style = "red" if data["result"].startswith("ERROR") else "dim"
        console.print(f"   [{style}]{preview}[/]")
    elif kind == "loop_warning":
        console.print(f"[yellow]Warned model: {data['name']} repeated x{data['count']}[/]")
    elif kind == "loop_detected":
        console.print(f"[yellow]Loop detected: {data['name']} x{data['count']}[/]")
    elif kind == "max_steps":
        console.print("[yellow]Hit MAX_STEPS[/]")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp1252 console chokes on "→"
    task = " ".join(sys.argv[1:]) or console.input("[bold]Task:[/] ")
    s = load_settings()
    llm = LLM(s.api_key, s.base_url, s.model)
    log = RunLog(meta={"model": s.model, "base_url": s.base_url, "task": task})

    def on_event(kind: str, data: dict):
        log(kind, data)
        show(kind, data)

    try:
        answer = run_agent(
            task, llm, default_registry(s.workspace), s.max_steps, on_event, s.loop_limit
        )
    except Exception as e:  # e.g. retries exhausted on 429: keep the evidence
        log("error", {"type": type(e).__name__, "message": str(e)})
        raise
    finally:
        console.print(f"[dim]log: {log.path}  tokens: {log.totals['total_tokens']}[/]")
    console.print(Panel(answer, title="Answer", border_style="green"))


if __name__ == "__main__":
    main()
