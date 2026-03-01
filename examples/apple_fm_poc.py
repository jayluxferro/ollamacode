#!/usr/bin/env python3
"""Apple FM integration POC.

Usage:
  OLLAMACODE_APPLE_FM_NATIVE_TOOLS=1 \
  OLLAMACODE_APPLE_FM_TRANSCRIPT_PATH=/tmp/ollamacode_apple_fm_transcript.jsonl \
  python3 examples/apple_fm_poc.py

Notes:
- Requires apple-fm-sdk and on-device model availability.
- If native tools are disabled, tool calls may be returned but not executed.
- Set OLLAMACODE_APPLE_FM_STATEFUL=1 to reuse a session between turns.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from ollamacode.providers.apple_fm_provider import AppleFMProvider


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer", "description": "First number"},
                    "b": {"type": "integer", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        },
    }
]


async def _executor(name: str, args: dict[str, Any]) -> str:
    if name == "add_numbers":
        return str(int(args.get("a", 0)) + int(args.get("b", 0)))
    return "unknown tool"


async def main() -> int:
    os.environ.setdefault(
        "OLLAMACODE_APPLE_FM_TRANSCRIPT_PATH",
        "/tmp/ollamacode_apple_fm_transcript.jsonl",
    )
    os.environ.setdefault("OLLAMACODE_APPLE_FM_STATEFUL", "1")
    provider = AppleFMProvider()
    provider.set_tool_executor(_executor)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Add 2 and 3 using the add_numbers tool, then answer briefly.",
        },
    ]

    try:
        resp = await provider.chat_async("apple.system", messages, TOOLS)
    except Exception as e:  # noqa: BLE001
        print(f"Apple FM error: {e}", file=sys.stderr)
        return 1

    msg = resp.get("message") if isinstance(resp, dict) else None
    if msg:
        print("Response:")
        print(msg)
    else:
        print(resp)

    # Second turn to validate stateful context.
    messages.append({"role": "assistant", "content": msg.get("content", "") if msg else ""})
    messages.append(
        {
            "role": "user",
            "content": "What number did you just compute? Answer with a single number.",
        }
    )
    resp2 = await provider.chat_async("apple.system", messages, TOOLS)
    msg2 = resp2.get("message") if isinstance(resp2, dict) else None
    if msg2:
        print("\nFollow-up:")
        print(msg2)
    else:
        print(resp2)

    print("\nTranscript path:")
    print(os.environ.get("OLLAMACODE_APPLE_FM_TRANSCRIPT_PATH"))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
