"""Run log: every agent event appended as one JSON line to runs/<timestamp>.jsonl.

It is just another `on_event` sink, so the loop doesn't know it exists.
Read a run back with e.g.:  uv run python -c "import json,sys; [print(json.loads(l)['kind']) for l in open(sys.argv[1])]" runs/X.jsonl
"""

import json
import time
from datetime import datetime
from pathlib import Path


class RunLog:
    def __init__(self, runs_dir: Path | str = "runs", meta: dict | None = None):
        runs_dir = Path(runs_dir)
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = runs_dir / f"{datetime.now():%Y%m%d-%H%M%S-%f}.jsonl"
        self.t0 = time.perf_counter()
        self.totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if meta:
            self("meta", meta)

    def __call__(self, kind: str, data: dict):
        if kind == "llm_call" and data.get("usage"):
            for k in self.totals:
                self.totals[k] += data["usage"].get(k) or 0
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            "t": round(time.perf_counter() - self.t0, 3),  # seconds since run start
            "kind": kind,
            **data,
        }
        if kind in ("final", "max_steps", "loop_detected", "error"):
            record["total_usage"] = self.totals
        # Open/append/close per event: a crash mid-run still leaves a readable log
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
