"""Tool registry: name -> (function, pydantic args model, description)."""

import json
from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., str]

    def usage(self) -> str:
        """One-line reminder of the expected args, appended to argument errors."""
        params = self.args_model.model_json_schema()
        required = set(params.get("required", []))
        parts = [
            f'"{k}": {v.get("type", "any")}{" (required)" if k in required else ""}'
            for k, v in params.get("properties", {}).items()
        ]
        return f"Expected JSON object: {{{', '.join(parts)}}}"

    def schema(self) -> dict:
        """OpenAI-style function schema."""
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, name: str, description: str, args_model: type[BaseModel]):
        def deco(fn):
            self.tools[name] = Tool(name, description, args_model, fn)
            return fn

        return deco

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools.values()]

    def run(self, name: str, raw_args: str) -> str:
        """Run a tool. Errors are returned as text so the model can self-correct."""
        tool = self.tools.get(name)
        if tool is None:
            return f"ERROR: unknown tool '{name}'. Available: {', '.join(self.tools)}"
        try:
            parsed = json.loads(raw_args or "{}")
        except json.JSONDecodeError as e:
            return f"ERROR: arguments for {name} are not valid JSON ({e}). {tool.usage()}"
        if not isinstance(parsed, dict):  # e.g. ["README.md"] or null
            return (
                f"ERROR: arguments for {name} must be a JSON object, "
                f"got {type(parsed).__name__}. {tool.usage()}"
            )
        problems = [f"unknown argument '{k}'" for k in parsed if k not in tool.args_model.model_fields]
        try:
            args = tool.args_model(**parsed)
        except ValidationError as e:
            # Short "field: message" lines; pydantic's full text has URLs and class names
            problems += [f"'{'.'.join(map(str, err['loc']))}': {err['msg']}" for err in e.errors()]
        if problems:
            return f"ERROR: invalid arguments for {name}: {'; '.join(problems)}. {tool.usage()}"
        try:
            return str(tool.fn(**args.model_dump()))
        except Exception as e:  # tool failures go back to the model, not crash the loop
            return f"ERROR: {type(e).__name__}: {e}"


def default_registry(workspace) -> Registry:
    from tools import files, shell

    reg = Registry()
    files.register(reg, workspace)
    shell.register(reg, workspace)
    return reg
