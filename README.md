# mini-harness

A from-scratch agent harness (loop + tools + retry) on Gemini Flash, for learning
how tools like Claude Code work. See `CLAUDE.md` for context, decisions and roadmap.

```bash
uv sync
cp .env.example .env    # fill GEMINI_API_KEY and MODEL
uv run python main.py "list the files here and summarize README.md"
uv run pytest
```

Continue in Claude Code: open this folder and run `claude`, then e.g.
"Read CLAUDE.md and do roadmap step 2".
