# mini-harness

A minimal AI agent harness written from scratch, for learning. The goal is to
understand what tools like Claude Code do under the hood (agent loop, tools,
retries, context, subagents), not to replace them.

## Background (from the planning chat)

- Owner: Long. Company uses Claude Code; this project is for learning harness
  engineering so he can use Claude Code better (CLAUDE.md, subagents, skills,
  hooks) and debug it when it misbehaves.
- Model: Google Gemini **Flash** on the free tier, called through Gemini's
  OpenAI-compatible endpoint. Key from https://aistudio.google.com/apikey.
  Copy the exact model ID from https://ai.google.dev/gemini-api/docs/models
  (prefer stable over preview models; preview models have tighter limits).
- Free-tier facts to respect: limits are per project (not per key), daily
  quota resets at midnight Pacific time, overruns return `429
  RESOURCE_EXHAUSTED`, and free-tier data may be used by Google to improve
  products — **never send company or client data** (SAP, etc.).
- Later options already discussed: OpenRouter `:free` models or Groq as a
  fallback provider; local models via Ollama (raise context: Ollama's default
  is tiny); 9router as a gateway once the hand-written failover is understood.

## Decisions

- **No agent framework at first** (no LangChain/LangGraph). The loop in
  `agent.py` is the thing being learned. Compare with LangGraph later.
- Use the `openai` SDK everywhere so switching provider = changing
  `BASE_URL` + `MODEL` in `.env`.
- Write retry/backoff by hand in `llm.py` (no tenacity).
- Tools are plain functions with a pydantic args model, registered in
  `tools/__init__.py`.

## Stack

Python 3.12+, uv, openai, pydantic, python-dotenv, rich, pytest.

```bash
uv sync                      # install
cp .env.example .env         # then fill GEMINI_API_KEY and MODEL
uv run python main.py "list the files here and summarize README.md"
uv run pytest                # tests use a fake LLM, no network
```

## Layout

- `config.py` — settings from `.env`
- `llm.py` — thin client: chat call + retry with exponential backoff on 429 / connection errors
- `tools/` — registry + `files.py` (read/write/list, sandboxed to WORKSPACE) and `shell.py` (whitelisted commands, no shell=True)
- `agent.py` — the loop: call model → run tool calls → append results → repeat until no tool calls, `MAX_STEPS`, or a loop (same tool+args+result `LOOP_LIMIT` times)
- `runlog.py` — `on_event` sink that appends every event to `runs/<timestamp>.jsonl` (messages, tool calls, latency, tokens)
- `main.py` — CLI with rich output; prints the log path + total tokens at the end
- `tests/` — loop tests with a scripted fake LLM

## Roadmap (do in order)

1. [x] Get a real run working against Gemini Flash.
2. [x] Log every turn to `runs/*.jsonl` (messages, tool calls, latency, tokens).
3. [x] Loop detection: stop if the same tool+args repeats N times.
4. [ ] Feed tool/JSON errors back to the model so it self-corrects.
5. [ ] Provider failover (Gemini → OpenRouter/Groq) on repeated 429.
6. [ ] Context management: truncate/summarize old tool results.
7. [ ] `spawn_agent` tool = subagent with a fresh context, returns a summary only.
8. [ ] MCP client support.
9. [ ] Small eval set (tasks + expected outcomes) to compare models/prompts.
10. [ ] Rebuild the same agent in LangGraph and compare.

## Conventions

- Keep files small and readable; this is a learning codebase.
- Every new tool gets a test.
- Explanations to Long: practical and builder-oriented; diagrams welcome.
