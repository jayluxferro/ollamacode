# OpenCode Parity Analysis

This document compares OllamaCode against the reference implementation in `tmp/opencode`.
It is intentionally scoped to coding-agent behavior, tool surface, session model, and runtime UX.

Last reviewed: 2026-03-06

## Summary

OllamaCode is already ahead of OpenCode in several backend areas:

- stronger filesystem and terminal hardening
- broader built-in MCP surface for git, checkpoints, semantic search, coverage, and provider support
- richer Python-side test coverage and simpler local-first deployment story

OllamaCode is now at parity on these key OpenCode features:

- session-scoped todos via `todoread` / `todowrite`
- first-class interactive `question` tool in CLI, rich TUI, Textual TUI, and protocol chat continuation
- first-class `task` delegation in CLI, rich TUI, Textual TUI, and protocol chat

The main remaining parity gaps are now mostly around transport polish and the unfinished alternative UI path:

- a more polished default interactive UI path than the current partially-finished Textual rewrite
- control-plane style remote workspace/session management

## Feature Matrix

| Capability | OpenCode | OllamaCode | Status |
|---|---|---|---|
| Read / write / edit / patch files | Yes | Yes (`fs_mcp`) | Parity |
| Glob / grep / code search | Yes | Yes (`codebase_mcp`) | Parity |
| Web fetch / web search | Yes | Yes | Parity |
| Skills | Yes | Yes | Parity |
| Session todos | Yes | Yes | Parity |
| Apply patch tool | Yes | Yes | Parity |
| LSP support | Yes | Yes | Parity |
| Multi-agent planning / review | Yes | Yes | Parity |
| Session export / import | Yes | Yes | Parity |
| Git operations | Partial | Stronger built-in support | Better |
| Checkpoints / rewind | Snapshot-based | Built-in session checkpoints | Better |
| Local provider variety | Strong | Stronger local-first Python provider stack | Better |
| Security hardening | Good | Stronger sandbox and validation layers | Better |
| `task` delegation tool | Yes | Yes in interactive clients and protocol chat; not yet complete in serve/chatStream | Near parity |
| Structured `question` tool | Yes | Yes in interactive clients; not yet complete in serve/chatStream | Near parity |
| Interactive permission queue | Yes | Yes for CLI, rich TUI, protocol, and HTTP chat continue | Near parity |
| Control plane / workspace proxy / PTY infra | Yes | Partial serve/protocol coverage only | Behind |

## What Changed In This Repo

Recent parity work completed in OllamaCode:

- session TODO persistence in SQLite
- Textual TODO sidebar synchronization
- `todoread` / `todowrite` MCP server with session scoping
- automatic session bootstrap for interactive chats so session-scoped tools are usable immediately
- stronger system-prompt guidance to actually use todo tracking on multi-step tasks

## Highest-Value Remaining Gaps

### 1. Finish or de-risk the Textual rewrite

The Textual UI has useful building blocks, but it is not yet the obvious best interface.
It still has gaps like stylesheet issues and missing orchestration features.

## Recommended Order

1. Finish wiring the Textual rewrite into the supported import/runtime path and stabilize its tests.
2. Extend `task` / `question` parity to serve/chatStream and other non-interactive transports.
3. Decide whether a control-plane/workspace-proxy layer is actually in scope for OllamaCode.

## Bar For “Better Than OpenCode”

OllamaCode should be considered clearly better once it has:

- OpenCode-level `task` and `question` tooling
- OpenCode-level permission workflow
- the current OllamaCode strengths preserved:
  - stronger security defaults
  - stronger checkpointing
  - stronger local-provider flexibility
  - stronger test coverage
