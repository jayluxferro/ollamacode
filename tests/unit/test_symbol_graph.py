from ollamacode.symbol_graph import build_symbol_graph
from ollamacode.symbol_index import build_symbol_index, query_symbol, find_references


def test_symbol_graph_basic(tmp_path) -> None:
    (tmp_path / "a.py").write_text(
        "def foo():\n    return 1\n\nfoo()\n"
    )
    graph = build_symbol_graph(str(tmp_path), max_files=10)
    assert "foo" in graph["definitions"]
    assert "a.py" in graph["callers"].get("foo", [])


def test_symbol_index_basic(tmp_path) -> None:
    (tmp_path / "b.py").write_text("class Bar:\n    pass\n\nBar()\n")
    db = tmp_path / "symbols.db"
    info = build_symbol_index(str(tmp_path), db_path=db, max_files=10)
    assert info["symbols"] >= 1
    defs = query_symbol("Bar", workspace_root=str(tmp_path), db_path=db)
    assert defs and defs[0]["path"] == "b.py"
    refs = find_references("Bar", workspace_root=str(tmp_path), db_path=db)
    assert refs
