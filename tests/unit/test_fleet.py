from __future__ import annotations

import pytest

from ollamacode.fleet import collect_fleet_snapshot


@pytest.mark.asyncio
async def test_collect_fleet_snapshot_local_only():
    snapshot = await collect_fleet_snapshot(
        [
            {
                "id": "w1",
                "name": "Local",
                "type": "local",
                "workspace_root": "/tmp/work",
            }
        ]
    )
    assert snapshot["total"] == 1
    assert snapshot["healthy"] == 1
    assert snapshot["workspaces"][0]["live"]["workspace_root"] == "/tmp/work"
