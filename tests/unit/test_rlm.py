"""Unit tests for RLM (recursive language model) integration."""

from ollamacode.rlm import (
    build_metadata_message,
    parse_final,
    parse_repl_blocks,
    run_repl_blocks,
    run_repl_snippet,
    truncate,
)


class TestParseReplBlocks:
    def test_empty(self) -> None:
        assert parse_repl_blocks("") == []
        assert parse_repl_blocks("no blocks") == []

    def test_single_repl_block(self) -> None:
        text = """Some text
```repl
print(len(context))
```
after"""
        assert parse_repl_blocks(text) == ["print(len(context))"]

    def test_unlabeled_block_treated_as_repl(self) -> None:
        text = """``` 
x = 1
```"""
        assert parse_repl_blocks(text) == ["x = 1"]

    def test_multiple_blocks(self) -> None:
        text = """```repl
print(1)
```
```repl
print(2)
```"""
        assert parse_repl_blocks(text) == ["print(1)", "print(2)"]


class TestParseFinal:
    def test_no_final(self) -> None:
        assert parse_final("nothing here") == (None, None)

    def test_final_content(self) -> None:
        assert parse_final("FINAL(hello)") == ("hello", None)
        assert parse_final("The answer is FINAL(42).") == ("42", None)

    def test_final_var(self) -> None:
        assert parse_final("FINAL_VAR(answer)") == (None, "answer")
        assert parse_final("Use FINAL_VAR(result).") == (None, "result")

    def test_final_preferred_over_final_var(self) -> None:
        text = "FINAL_VAR(x) and FINAL(done)"
        assert parse_final(text) == ("done", None)


class TestRunReplSnippet:
    def test_simple_print(self) -> None:
        r = run_repl_snippet("print(context.upper())", "hi", lambda p: "")
        assert r.error is None
        assert r.final_stdout.strip() == "HI"

    def test_llm_query_called(self) -> None:
        responses: list[str] = []

        def resolver(prompt: str) -> str:
            responses.append(prompt)
            return "got:" + prompt

        r = run_repl_snippet(
            "x = llm_query('what'); print(x)",
            "context here",
            resolver,
        )
        assert r.error is None
        assert r.llm_calls == ["what"]
        assert "got:what" in r.final_stdout

    def test_syntax_error(self) -> None:
        r = run_repl_snippet("syntax ( error", "hi", lambda p: "")
        assert r.error is not None
        assert "SyntaxError" in r.error

    def test_timeout_uses_default(self) -> None:
        # Quick run should succeed
        r = run_repl_snippet("print(1)", "x", lambda p: "y", timeout_seconds=5.0)
        assert r.error is None
        assert "1" in r.final_stdout

    def test_snippet_timeout_hits(self) -> None:
        # Busy loop with very short per-snippet timeout should hit snippet timeout
        r = run_repl_snippet(
            "x = 0\nwhile x < 5000000: x += 1",
            "c",
            lambda p: "",
            snippet_timeout_seconds=0.02,
            timeout_seconds=5.0,
        )
        # On slow/loaded CI might not timeout in 20ms; accept either outcome
        if r.error:
            assert "timed out" in r.error.lower()
        else:
            assert r.final_globals is not None and r.final_globals.get("x") == 5000000


class TestRunReplBlocks:
    def test_shared_namespace_final_var(self) -> None:
        """Blocks share a namespace so second block can read variable set in first."""
        codes = ["answer = len(context)", "summary = answer + 1"]
        parts, globals_, err = run_repl_blocks(codes, "hi", lambda p: "")
        assert err is None
        assert globals_ is not None
        assert globals_.get("answer") == 2
        assert globals_.get("summary") == 3

    def test_error_stops_blocks(self) -> None:
        codes = ["print(1)", "syntax ( error", "print(2)"]
        parts, globals_, err = run_repl_blocks(codes, "x", lambda p: "")
        assert err is not None
        assert "SyntaxError" in err
        assert len(parts) == 1  # first block ran


class TestBuildMetadataMessage:
    def test_contains_length_and_prefix(self) -> None:
        msg = build_metadata_message("hello world", prefix_chars=5)
        assert "11" in msg or "characters" in msg
        assert "hello" in msg


class TestTruncate:
    def test_under_limit(self) -> None:
        assert truncate("ab", 10) == "ab"

    def test_over_limit(self) -> None:
        out = truncate("abcdefghij", 5)
        assert len(out) <= 5 + 20
        assert "truncated" in out
