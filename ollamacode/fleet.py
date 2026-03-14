"""Helpers for aggregating workspace fleet status."""

from __future__ import annotations

from typing import Any


async def collect_fleet_snapshot(workspaces: list[dict[str, Any]]) -> dict[str, Any]:
    """Collect a live fleet snapshot from local registry entries."""
    import httpx

    snapshot_rows: list[dict[str, Any]] = []
    healthy = 0
    unhealthy = 0
    remote = 0

    for workspace in workspaces:
        row = dict(workspace)
        if row.get("type") != "remote" or not row.get("base_url"):
            row["live"] = {
                "ok": True,
                "status_code": 200,
                "session_count": None,
                "workspace_root": row.get("workspace_root") or "",
            }
            snapshot_rows.append(row)
            healthy += 1
            continue

        remote += 1
        base_url = str(row.get("base_url") or "").rstrip("/")
        headers: dict[str, str] = {}
        if row.get("api_key"):
            headers["Authorization"] = f"Bearer {row['api_key']}"
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                health = await client.get(base_url + "/health", headers=headers)
                workspace_info = await client.get(
                    base_url + "/workspace", headers=headers
                )
            ok = health.status_code == 200 and workspace_info.status_code == 200
            info_payload = (
                workspace_info.json()
                if workspace_info.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else {}
            )
            row["live"] = {
                "ok": ok,
                "status_code": workspace_info.status_code,
                "session_count": info_payload.get("sessionCount"),
                "workspace_root": info_payload.get("workspaceRoot"),
            }
            if ok:
                healthy += 1
            else:
                unhealthy += 1
        except Exception as exc:
            row["live"] = {
                "ok": False,
                "status_code": 0,
                "session_count": None,
                "workspace_root": None,
                "error": str(exc),
            }
            unhealthy += 1
        snapshot_rows.append(row)

    return {
        "total": len(workspaces),
        "remote": remote,
        "healthy": healthy,
        "unhealthy": unhealthy,
        "workspaces": snapshot_rows,
    }
