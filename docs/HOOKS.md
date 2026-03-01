# Hooks

OllamaCode supports lightweight pre/post tool hooks (similar to Claude Code).

## Config

Create a JSON file:

- `~/.ollamacode/hooks.json` (user)
- `.ollamacode/hooks.json` (project)

Example:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "write_file|edit_file|multi_edit",
        "hooks": [
          {
            "type": "command",
            "command": "python scripts/check_policy.py",
            "timeout": 20
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": ".*",
        "hooks": [
          {
            "type": "http",
            "url": "http://localhost:8080/hook"
          }
        ]
      }
    ]
  }
}
```

## Hook payload

PreToolUse:

```json
{
  "event": "PreToolUse",
  "toolName": "write_file",
  "toolInput": {"path": "README.md", "content": "..."},
  "workspaceRoot": "/path/to/repo",
  "sessionId": "uuid-or-empty",
  "userPrompt": "user prompt text",
  "timestamp": 1730000000.0
}
```

PostToolUse adds `toolOutput` and `isError`.

## Hook response

Return JSON on stdout (command) or in HTTP response body:

```json
{
  "decision": {
    "behavior": "allow|deny|modify",
    "message": "optional reason",
    "updatedInput": {"path": "README.md", "content": "..." }
  }
}
```

Notes:
- `deny` skips the tool call.
- `modify` replaces arguments.
- `allow` skips any confirmation prompt (when enabled).

Disable hooks with `OLLAMACODE_DISABLE_HOOKS=1`.
