"""Unit tests for structured apply-edits (<<EDITS>> JSON)."""

from ollamacode.edits import apply_edits, parse_edits


def test_parse_edits_empty():
    assert parse_edits("no markers") == []
    assert parse_edits("<<EDITS>>\n<<END>>") == []


def test_parse_edits_single():
    text = 'Here is the fix.\n<<EDITS>>\n[{"path": "a.py", "newText": "x = 1"}]\n<<END>>'
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
