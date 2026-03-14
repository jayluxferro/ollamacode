from __future__ import annotations

from ollamacode.workspaces import (
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
)


def test_workspace_registry_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ollamacode.workspaces._WORKSPACES_PATH", tmp_path / "workspaces.json"
    )
    workspace = create_workspace(
        name="Remote Dev",
        kind="remote",
        base_url="http://localhost:9000",
        api_key="secret",
    )
    assert get_workspace(workspace["id"])["name"] == "Remote Dev"
    assert len(list_workspaces()) == 1
    assert delete_workspace(workspace["id"]) is True
    assert get_workspace(workspace["id"]) is None
