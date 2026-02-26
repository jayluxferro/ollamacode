from pathlib import Path

from ollamacode.refactor import rename_symbol, extract_function, move_function


def test_rename_symbol(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return foo()\n")
    edits = rename_symbol(str(tmp_path), "foo", "bar")
    assert edits


def test_extract_function(tmp_path: Path) -> None:
    path = tmp_path / "b.py"
    path.write_text("a = 1\nb = 2\nc = a + b\n")
    out = extract_function(str(path), 1, 2, "init_vals")
    assert out
    assert "def init_vals()" in path.read_text()


def test_move_function(tmp_path: Path) -> None:
    src = tmp_path / "src.py"
    dst = tmp_path / "dst.py"
    src.write_text("def foo():\n    return 1\n\nx = 3\n")
    ok = move_function(str(src), "foo", str(dst))
    assert ok
    assert "def foo" not in src.read_text()
    assert "def foo" in dst.read_text()
