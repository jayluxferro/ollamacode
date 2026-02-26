from ollamacode import vector_memory as vm


def test_embed_many_provider_batch(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_batch(texts, config):
        calls.append(list(texts))
        return [[1.0] for _ in texts]

    monkeypatch.setattr(vm, "_embed_via_provider_many", fake_batch)
    out = vm._embed_many(["a", "b"], {"embedding_backend": "provider"})
    assert calls == [["a", "b"]]
    assert out == [[1.0], [1.0]]


def test_build_vector_index_batches(monkeypatch, tmp_path) -> None:
    (tmp_path / "a.md").write_text("# T\n\nhello world\n")
    calls: list[list[str]] = []

    def fake_embed_many(texts, config):
        calls.append(list(texts))
        return [[0.1, 0.2, 0.3] for _ in texts]

    monkeypatch.setattr(vm, "_embed_many", fake_embed_many)
    monkeypatch.setattr(vm, "_VM_DB_PATH", tmp_path / "vm.db")

    info = vm.build_vector_index(str(tmp_path), max_files=10, embed=True, config={})
    assert info["indexed_files"] >= 1
    assert calls and len(calls[0]) >= 1
