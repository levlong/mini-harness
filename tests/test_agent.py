"""Loop tests with a scripted fake LLM — no network, no API key."""

import json
from types import SimpleNamespace as NS

from agent import run_agent
from tools import default_registry


def tool_call(id_, name, args):
    return NS(id=id_, function=NS(name=name, arguments=json.dumps(args)))


class FakeLLM:
    """Returns pre-scripted assistant messages in order, records what it was sent."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        return self.script.pop(0)


def test_reads_file_then_answers(tmp_path):
    (tmp_path / "README.md").write_text("hello harness")
    llm = FakeLLM(
        [
            NS(
                content=None,
                tool_calls=[tool_call("c1", "read_file", {"path": "README.md"})],
            ),
            NS(content="README says hello harness", tool_calls=None),
        ]
    )
    answer = run_agent("summarize README", llm, default_registry(tmp_path))
    assert answer == "README says hello harness"
    tool_msg = llm.calls[1][-1]
    assert tool_msg["role"] == "tool" and tool_msg["content"] == "hello harness"


def test_tool_error_goes_back_to_model(tmp_path):
    llm = FakeLLM(
        [
            NS(
                content=None,
                tool_calls=[tool_call("c1", "read_file", {"path": "../secret"})],
            ),
            NS(content="cannot read it", tool_calls=None),
        ]
    )
    run_agent("read secret", llm, default_registry(tmp_path))
    assert llm.calls[1][-1]["content"].startswith("ERROR: PermissionError")


def test_unknown_tool_and_bad_args(tmp_path):
    reg = default_registry(tmp_path)
    assert reg.run("nope", "{}").startswith("ERROR: unknown tool")
    assert reg.run("read_file", "{not json").startswith("ERROR: invalid arguments")


def test_shell_whitelist(tmp_path):
    reg = default_registry(tmp_path)
    assert reg.run("run_command", json.dumps({"command": "rm -rf ."})).startswith(
        "ERROR"
    )
    assert "exit=0" in reg.run("run_command", json.dumps({"command": "echo hi"}))


def test_stops_at_max_steps(tmp_path):
    loop = [
        NS(content=None, tool_calls=[tool_call(f"c{i}", "list_dir", {"path": "."})])
        for i in range(3)
    ]
    answer = run_agent(
        "loop forever",
        FakeLLM(loop),
        default_registry(tmp_path),
        max_steps=3,
        loop_limit=99,
    )
    assert answer.startswith("Stopped after 3 steps")


def test_gemini_thought_signature_is_preserved():
    """Real SDK object with Gemini's extra field -> must survive into the next request."""
    from openai.types.chat import ChatCompletionMessage

    from agent import assistant_to_dict

    msg = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_dir", "arguments": '{"path": "."}'},
                    "extra_content": {"google": {"thought_signature": "SIG123"}},
                }
            ],
        }
    )
    d = assistant_to_dict(msg)
    assert (
        d["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "SIG123"
    )
    assert d["content"] == "" and d["role"] == "assistant"


def test_run_log_writes_jsonl(tmp_path):
    from runlog import RunLog

    (tmp_path / "a.txt").write_text("hi")
    first = NS(content=None, tool_calls=[tool_call("c1", "read_file", {"path": "a.txt"})])
    llm = FakeLLM([first, NS(content="done", tool_calls=None)])
    llm.last_usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

    log = RunLog(tmp_path / "runs", meta={"model": "fake"})
    run_agent("read a.txt", llm, default_registry(tmp_path), on_event=log)

    records = [json.loads(line) for line in log.path.read_text().splitlines()]
    kinds = [r["kind"] for r in records]
    assert kinds == ["meta", "start", "llm_call", "tool_call", "tool_result", "llm_call", "final"]
    assert records[1]["messages"][1]["content"] == "read a.txt"
    assert records[2]["latency_s"] >= 0 and records[2]["usage"]["total_tokens"] == 15
    assert records[4]["result"] == "hi" and records[4]["id"] == "c1"
    assert records[-1]["total_usage"]["total_tokens"] == 30


def test_loop_detected_on_repeated_call(tmp_path):
    # same tool + same args (key order differs) + same result, 3 times -> stop
    args = ['{"path": "."}', '{"path":"."}', '{ "path" : "." }']
    script = [
        NS(content=None, tool_calls=[NS(id=f"c{i}", function=NS(name="list_dir", arguments=a))])
        for i, a in enumerate(args)
    ]
    events = []
    answer = run_agent(
        "list",
        FakeLLM(script),
        default_registry(tmp_path),
        on_event=lambda k, d: events.append(k),
        loop_limit=3,
    )
    assert answer.startswith("Stopped: list_dir") and "loop detected" in answer
    assert events.count("loop_warning") == 1 and events[-1] == "loop_detected"


def test_same_call_with_new_result_is_not_a_loop(tmp_path):
    # re-reading a file that changed between reads is progress, not a loop
    f = tmp_path / "a.txt"
    reads = [NS(content=None, tool_calls=[tool_call(f"c{i}", "read_file", {"path": "a.txt"})]) for i in range(3)]

    class ChangingFile(FakeLLM):
        def chat(self, messages, tools=None):
            f.write_text(f"version {len(self.calls)}")
            return super().chat(messages, tools)

    llm = ChangingFile(reads + [NS(content="done", tool_calls=None)])
    assert run_agent("watch a.txt", llm, default_registry(tmp_path), loop_limit=3) == "done"


def test_call_key_normalizes_arg_order():
    from agent import call_key

    assert call_key("t", '{"a":1,"b":2}', "r") == call_key("t", '{"b": 2, "a": 1}', "r")
    assert call_key("t", "{bad json", "r") == ("t", "{bad json", "r")


def test_loop_warning_lets_model_recover(tmp_path):
    # 2nd identical call gets a warning appended; the model listens and answers
    ls = [NS(content=None, tool_calls=[tool_call(f"c{i}", "list_dir", {"path": "."})]) for i in range(2)]
    llm = FakeLLM(ls + [NS(content="it is empty", tool_calls=None)])
    answer = run_agent("list", llm, default_registry(tmp_path), loop_limit=3)
    assert answer == "it is empty"
    first, second = llm.calls[1][-1]["content"], llm.calls[2][-1]["content"]
    assert "WARNING" not in first
    assert second.startswith("(empty)") and "WARNING" in second
