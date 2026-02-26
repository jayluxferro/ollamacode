from ollamacode.repo_map import build_repo_map
from ollamacode.repo_map import build_symbol_index


def test_build_repo_map_basic(tmp_path) -> None:
    (tmp_path / "a.py").write_text("def foo():\n    return 1\n")
    (tmp_path / "b.ts").write_text("export function bar() { return 2 }\n")
    out = build_repo_map(str(tmp_path), max_files=10)
    assert "Repo Map" in out
    assert "a.py" in out
    assert "b.ts" in out
    assert "foo" in out
    assert "bar" in out


def test_build_symbol_index(tmp_path) -> None:
    (tmp_path / "c.py").write_text("class Thing:\n    pass\n")
    out = build_symbol_index(str(tmp_path), max_files=10)
    assert "c.py" in out
    assert "Thing" in out["c.py"]
