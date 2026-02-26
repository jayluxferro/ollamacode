"""Unit tests for structured apply-edits (<<EDITS>> JSON), parse_reasoning, parse_review."""

from ollamacode.edits import (
    apply_edits,
    parse_edits,
    parse_reasoning,
    parse_review,
    apply_unified_diff_filtered,
)


def test_parse_edits_empty():
    assert parse_edits("no markers") == []
    assert parse_edits("<<EDITS>>\n<<END>>") == []


def test_parse_edits_single():
    text = (
        'Here is the fix.\n<<EDITS>>\n[{"path": "a.py", "newText": "x = 1"}]\n<<END>>'
    )
    got = parse_edits(text)
    assert len(got) == 1
    assert got[0]["path"] == "a.py"
    assert got[0]["newText"] == "x = 1"
    assert got[0]["oldText"] is None


def test_parse_edits_with_old_text():
    text = '<<EDITS>>\n[{"path": "b.py", "oldText": "old", "newText": "new"}]\n<<END>>'
    got = parse_edits(text)
    assert len(got) == 1
    assert got[0]["oldText"] == "old"
    assert got[0]["newText"] == "new"


def test_apply_edits_full_file(tmp_path):
    edits = [{"path": "out.txt", "newText": "hello"}]
    n = apply_edits(edits, tmp_path)
    assert n == 1
    assert (tmp_path / "out.txt").read_text() == "hello"


def test_apply_edits_search_replace(tmp_path):
    (tmp_path / "f.py").write_text("a = 1\na = 2\n")
    edits = [{"path": "f.py", "oldText": "a = 1", "newText": "a = 99"}]
    n = apply_edits(edits, tmp_path)
    assert n == 1
    assert (tmp_path / "f.py").read_text() == "a = 99\na = 2\n"


def test_parse_reasoning_missing():
    """parse_reasoning returns (None, text) when no block."""
    out = parse_reasoning("just some text")
    assert out[0] is None
    assert out[1] == "just some text"


def test_parse_reasoning_present():
    """parse_reasoning extracts steps and conclusion."""
    text = (
        'Hi.\n<<REASONING>>\n{"steps": ["a", "b"], "conclusion": "done"}\n<<END>>\nBye.'
    )
    reasoning, rest = parse_reasoning(text)
    assert reasoning is not None
    assert reasoning["steps"] == ["a", "b"]
    assert reasoning["conclusion"] == "done"
    assert "Bye" in rest and "REASONING" not in rest


def test_parse_review_missing():
    """parse_review returns (None, text) when no block."""
    out = parse_review("no review block")
    assert out[0] is None
    assert out[1] == "no review block"


def test_parse_review_present():
    """parse_review extracts suggestions."""
    text = '<<REVIEW>>\n{"suggestions": [{"location": "f:1", "suggestion": "add type", "rationale": "clarity"}]}\n<<END>>'
    sugs, rest = parse_review(text)
    assert sugs is not None
    assert len(sugs) == 1
    assert sugs[0]["location"] == "f:1"
    assert sugs[0]["suggestion"] == "add type"
    assert sugs[0]["rationale"] == "clarity"


def test_apply_edits_outside_workspace(tmp_path):
    """Edits with path outside workspace_root are skipped (write scope)."""
    edits = [
        {"path": "ok.txt", "newText": "ok"},
        {"path": "../outside.txt", "newText": "bad"},
        {"path": "/etc/passwd", "newText": "bad"},
    ]
    n = apply_edits(edits, tmp_path)
    assert n == 1
    assert (tmp_path / "ok.txt").read_text() == "ok"


def test_apply_edits_unified_diff_fuzzy(tmp_path):
    (tmp_path / "foo.txt").write_text("line1\nline2\nline3\n")
    diff = "\n".join(
        [
            "diff --git a/foo.txt b/foo.txt",
            "--- a/foo.txt",
            "+++ b/foo.txt",
            "@@ -5,2 +5,2 @@",
            " line2",
            "-line3",
            "+line3-mod",
            "",
        ]
    )
    edits = [{"path": "ignored.txt", "newText": diff}]
    n = apply_edits(edits, tmp_path)
    assert n == 1
    assert (tmp_path / "foo.txt").read_text() == "line1\nline2\nline3-mod"


def test_apply_edits_anchor_insert(tmp_path):
    path = tmp_path / "anchor.txt"
    path.write_text("alpha\nbeta\n")
    edits = [
        {
            "path": "anchor.txt",
            "newText": "X\n",
            "anchor": "alpha\n",
            "position": "after",
        }
    ]
    n = apply_edits(edits, tmp_path)
    assert n == 1
    assert path.read_text() == "alpha\nX\nbeta\n"


def test_apply_unified_diff_filtered(tmp_path):
    (tmp_path / "f.txt").write_text("a\nb\nc\n")
    diff = "\n".join(
        [
            "diff --git a/f.txt b/f.txt",
            "--- a/f.txt",
            "+++ b/f.txt",
            "@@ -1,1 +1,1 @@",
            "-a",
            "+a1",
            "@@ -3,1 +3,1 @@",
            "-c",
            "+c1",
            "",
        ]
    )

    def include(path, idx, h):
        return idx == 1

    n = apply_unified_diff_filtered(diff, tmp_path, include)
    assert n == 1
    assert (tmp_path / "f.txt").read_text() == "a\nb\nc1"
