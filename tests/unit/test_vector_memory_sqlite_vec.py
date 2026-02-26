import sqlite3

import pytest

import ollamacode.vector_memory as vm


@pytest.mark.skipif(
    not hasattr(vm, "_sqlite_vec_available"),
    reason="sqlite-vec check helper not present",
)
def test_sqlite_vec_optional(tmp_path) -> None:
    db = tmp_path / "vm.db"
    conn = sqlite3.connect(str(db))
    try:
        vm._try_enable_sqlite_vec(conn)
        # Always safe to call; just ensure it doesn't crash.
        if vm._sqlite_vec_available():
            vm._ensure_vec_table(conn, 3)
            vm._insert_vec(conn, 1, [0.1, 0.2, 0.3])
    finally:
        conn.close()
