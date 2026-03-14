const STORAGE_KEYS = {
  workspaceId: "ollamacode:selectedWorkspaceId",
  sessionId: "ollamacode:selectedSessionId",
};

const state = {
  selectedWorkspaceId: localStorage.getItem(STORAGE_KEYS.workspaceId),
  selectedSessionId: localStorage.getItem(STORAGE_KEYS.sessionId),
  eventSource: null,
  eventSourceUrl: "",
};

const els = {
  workspaceTitle: document.querySelector("#workspace-title"),
  workspaceMeta: document.querySelector("#workspace-meta"),
  workspaceList: document.querySelector("#workspace-list"),
  workspaceForm: document.querySelector("#workspace-form"),
  workspaceName: document.querySelector("#workspace-name"),
  workspaceUrl: document.querySelector("#workspace-url"),
  workspaceKey: document.querySelector("#workspace-key"),
  fleetSummary: document.querySelector("#fleet-summary"),
  principalForm: document.querySelector("#principal-form"),
  principalName: document.querySelector("#principal-name"),
  principalRole: document.querySelector("#principal-role"),
  principalList: document.querySelector("#principal-list"),
  refreshWorkspace: document.querySelector("#refresh-workspace"),
  sessionSearch: document.querySelector("#session-search"),
  sessionList: document.querySelector("#session-list"),
  activityLog: document.querySelector("#activity-log"),
  clearActivity: document.querySelector("#clear-activity"),
  sessionDetail: document.querySelector("#session-detail"),
  sessionBadge: document.querySelector("#session-badge"),
  importJson: document.querySelector("#import-json"),
  importSession: document.querySelector("#import-session"),
  newSession: document.querySelector("#new-session"),
  chatLog: document.querySelector("#chat-log"),
  chatForm: document.querySelector("#chat-form"),
  chatStream: document.querySelector("#chat-stream"),
  chatInput: document.querySelector("#chat-input"),
  clearChat: document.querySelector("#clear-chat"),
};

function currentApiBase() {
  return state.selectedWorkspaceId
    ? `/workspaces/${state.selectedWorkspaceId}/proxy`
    : "";
}

function persistSelection() {
  if (state.selectedWorkspaceId) {
    localStorage.setItem(STORAGE_KEYS.workspaceId, state.selectedWorkspaceId);
  } else {
    localStorage.removeItem(STORAGE_KEYS.workspaceId);
  }
  if (state.selectedSessionId) {
    localStorage.setItem(STORAGE_KEYS.sessionId, state.selectedSessionId);
  } else {
    localStorage.removeItem(STORAGE_KEYS.sessionId);
  }
}

async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "content-type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) return res.json();
  return { content: await res.text() };
}

function bubble(kind, text) {
  const div = document.createElement("div");
  div.className = `bubble ${kind}`;
  div.textContent = text;
  els.chatLog.appendChild(div);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function itemTemplate(title, subtitle, badgeText = "", badgeClass = "") {
  const div = document.createElement("div");
  div.className = "item";
  const badge = badgeText
    ? `<span class="badge ${badgeClass}">${badgeText}</span>`
    : "";
  div.innerHTML = `<h3>${title}</h3><p>${subtitle}</p>${badge}<div class="actions"></div>`;
  return div;
}

function detailBlock(title, body) {
  const div = document.createElement("div");
  div.className = "detail-block";
  div.innerHTML = `<h3>${title}</h3><pre>${body}</pre>`;
  return div;
}

function detailButtonList(title, rows, onOpen) {
  const div = document.createElement("div");
  div.className = "detail-block";
  const header = document.createElement("h3");
  header.textContent = title;
  div.appendChild(header);
  if (!rows.length) {
    const pre = document.createElement("pre");
    pre.textContent = `No ${title.toLowerCase()}`;
    div.appendChild(pre);
    return div;
  }
  const actions = document.createElement("div");
  actions.className = "actions";
  for (const row of rows) {
    const button = document.createElement("button");
    button.className = "secondary";
    button.textContent = `${row.id.slice(0, 8)} · ${row.title || "(untitled)"}`;
    button.onclick = () => onOpen(row.id);
    actions.appendChild(button);
  }
  div.appendChild(actions);
  return div;
}

function renderActivityItem(event) {
  const div = document.createElement("div");
  div.className = "activity-item";
  div.innerHTML = `<strong>${event.type}</strong><pre>${JSON.stringify(event.payload, null, 2)}</pre>`;
  return div;
}

function prependActivity(event) {
  if (!els.activityLog) return;
  els.activityLog.prepend(renderActivityItem(event));
  while (els.activityLog.children.length > 25) {
    els.activityLog.removeChild(els.activityLog.lastChild);
  }
}

async function loadWorkspaceInfo() {
  if (state.selectedWorkspaceId) {
    const workspace = await api(`/workspaces/${state.selectedWorkspaceId}`);
    const row = workspace.workspace;
    if (row) {
      els.workspaceTitle.textContent = row.name;
      els.workspaceMeta.textContent = `${row.base_url || row.workspace_root || row.type} · ${row.last_status || "unknown"} · ${row.owner || "unowned"} (${row.role || "owner"})`;
      return;
    }
  }
  const info = await api("/workspace");
  els.workspaceTitle.textContent = "Local Workspace";
  els.workspaceMeta.textContent = `${info.workspaceRoot} · ${info.sessionCount} sessions`;
}

async function loadFleetSummary() {
  const result = await api("/fleet");
  clearChildren(els.fleetSummary);
  const summary = detailBlock(
    "Fleet Status",
    `Total workspaces: ${result.total || 0}\nRemote: ${result.remote || 0}\nHealthy: ${result.healthy || 0}\nUnhealthy: ${result.unhealthy || 0}`,
  );
  els.fleetSummary.appendChild(summary);
  for (const workspace of result.workspaces || []) {
    const live = workspace.live || {};
    els.fleetSummary.appendChild(
      detailBlock(
        workspace.name || "Workspace",
        `${workspace.type}\n${live.ok ? "reachable" : "unreachable"}\nSessions: ${live.session_count ?? "?"}\nRoot: ${live.workspace_root ?? workspace.workspace_root ?? ""}`,
      ),
    );
  }
}

async function loadPrincipals() {
  const result = await api("/principals");
  clearChildren(els.principalList);
  for (const principal of result.principals || []) {
    const node = itemTemplate(
      principal.name,
      `${principal.role} · workspaces: ${(principal.workspace_ids || []).length}`,
    );
    const actions = node.querySelector(".actions");
    const edit = document.createElement("button");
    edit.className = "secondary";
    edit.textContent = "Edit";
    edit.onclick = async () => {
      const name = window.prompt("Principal name", principal.name) || principal.name;
      const role = window.prompt("Principal role", principal.role) || principal.role;
      const workspaceIDs = window
        .prompt(
          "Assigned workspace IDs (comma separated)",
          (principal.workspace_ids || []).join(","),
        )
        .split(",")
        .map((item) => item.trim())
        .filter(Boolean);
      await api(`/principals/${principal.id}`, {
        method: "PATCH",
        body: JSON.stringify({ name, role, workspaceIDs }),
      });
      await refreshAll();
    };
    actions.appendChild(edit);
    const remove = document.createElement("button");
    remove.className = "secondary";
    remove.textContent = "Delete";
    remove.onclick = async () => {
      await fetch(`/principals/${principal.id}`, { method: "DELETE" });
      await refreshAll();
    };
    actions.appendChild(remove);
    els.principalList.appendChild(node);
  }
}

async function loadWorkspaces() {
  const data = await api("/workspaces");
  clearChildren(els.workspaceList);
  for (const workspace of data.workspaces || []) {
    const active = workspace.id === state.selectedWorkspaceId;
    const node = itemTemplate(
      workspace.name,
      workspace.base_url || workspace.workspace_root || workspace.type,
      workspace.last_status || workspace.type,
      active ? "active" : workspace.last_status === "ok" ? "ok" : workspace.last_status === "error" ? "warn" : "",
    );
    const actions = node.querySelector(".actions");

    const select = document.createElement("button");
    select.className = active ? "" : "secondary";
    select.textContent = active ? "Selected" : "Select";
    select.onclick = async () => {
      state.selectedWorkspaceId = active ? null : workspace.id;
      state.selectedSessionId = null;
      persistSelection();
      await refreshAll();
    };
    actions.appendChild(select);

    const health = document.createElement("button");
    health.className = "secondary";
    health.textContent = "Health";
    health.onclick = async () => {
      const result = await api(`/workspaces/${workspace.id}/health`);
      bubble("system", `${workspace.name}: ${result.ok ? "healthy" : "unhealthy"}${result.error ? `\n${result.error}` : ""}`);
      await loadWorkspaces();
      await loadWorkspaceInfo();
    };
    actions.appendChild(health);

    const edit = document.createElement("button");
    edit.className = "secondary";
    edit.textContent = "Edit";
    edit.onclick = async () => {
      const name = window.prompt("Workspace name", workspace.name) || workspace.name;
      const baseUrl =
        window.prompt("Workspace base URL", workspace.base_url || "") ||
        workspace.base_url ||
        "";
      const owner =
        window.prompt("Workspace owner", workspace.owner || "") ||
        workspace.owner ||
        "";
      const role =
        window.prompt("Workspace role", workspace.role || "owner") ||
        workspace.role ||
        "owner";
      await api(`/workspaces/${workspace.id}`, {
        method: "PATCH",
        body: JSON.stringify({ name, baseUrl, owner, role }),
      });
      await refreshAll();
    };
    actions.appendChild(edit);

    const remove = document.createElement("button");
    remove.className = "secondary";
    remove.textContent = "Delete";
    remove.onclick = async () => {
      await fetch(`/workspaces/${workspace.id}`, { method: "DELETE" });
      if (state.selectedWorkspaceId === workspace.id) {
        state.selectedWorkspaceId = null;
        state.selectedSessionId = null;
      }
      persistSelection();
      await refreshAll();
    };
    actions.appendChild(remove);

    els.workspaceList.appendChild(node);
  }
}

async function loadSessions() {
  const query = els.sessionSearch.value.trim();
  const path = `${currentApiBase()}/sessions${query ? `?search=${encodeURIComponent(query)}` : ""}`;
  const data = await api(path);
  clearChildren(els.sessionList);
  for (const session of data.sessions || []) {
    const active = session.id === state.selectedSessionId;
    const node = itemTemplate(
      session.title || "(untitled)",
      `${session.id.slice(0, 8)} · ${session.message_count} messages`,
      active ? "active" : "",
      active ? "active" : "",
    );
    const actions = node.querySelector(".actions");

    const open = document.createElement("button");
    open.className = active ? "" : "secondary";
    open.textContent = active ? "Open" : "Open";
    open.onclick = async () => {
      state.selectedSessionId = session.id;
      persistSelection();
      await loadSessionDetail();
    };
    actions.appendChild(open);

    const branch = document.createElement("button");
    branch.className = "secondary";
    branch.textContent = "Branch";
    branch.onclick = async () => {
      const result = await api(`${currentApiBase()}/sessions/${session.id}/branch`, {
        method: "POST",
        body: JSON.stringify({}),
      });
      state.selectedSessionId = result.session.id;
      persistSelection();
      await refreshAll();
    };
    actions.appendChild(branch);

    const remove = document.createElement("button");
    remove.className = "secondary";
    remove.textContent = "Delete";
    remove.onclick = async () => {
      await fetch(`${currentApiBase()}/sessions/${session.id}`, { method: "DELETE" });
      if (state.selectedSessionId === session.id) state.selectedSessionId = null;
      persistSelection();
      await refreshAll();
    };
    actions.appendChild(remove);

    els.sessionList.appendChild(node);
  }
}

async function loadSessionDetail() {
  clearChildren(els.sessionDetail);
  if (!state.selectedSessionId) {
    els.sessionBadge.textContent = "No session";
    els.sessionBadge.className = "badge";
    els.sessionDetail.appendChild(detailBlock("Session", "Select a session to inspect details."));
    return;
  }
  els.sessionBadge.textContent = state.selectedSessionId.slice(0, 8);
  els.sessionBadge.className = "badge active";

  const timelineRes = await api(`${currentApiBase()}/sessions/${state.selectedSessionId}/timeline`);
  const timeline = timelineRes.timeline || {};
  const info = timeline.session || {};
  const messages = timeline.messages || [];
  const todos = timeline.todos || [];
  const checkpoints = timeline.checkpoints || [];
  const children = timeline.children || [];
  const ancestors = timeline.ancestors || [];

  els.sessionDetail.appendChild(
    detailBlock(
      info.title || "(untitled)",
      `${info.workspace_root || "workspace"}\n${messages.length} messages\nOwner: ${info.owner || "unowned"} (${info.role || "owner"})\nUpdated: ${info.updated_at || ""}`,
    ),
  );

  const sessionActions = document.createElement("div");
  sessionActions.className = "actions";

  const exportBtn = document.createElement("button");
  exportBtn.textContent = "Export";
  exportBtn.onclick = async () => {
    const result = await api(`${currentApiBase()}/sessions/${state.selectedSessionId}/export`);
    els.importJson.value = result.data || "";
    bubble("system", "Session export copied into the import box.");
  };
  sessionActions.appendChild(exportBtn);

  const renameBtn = document.createElement("button");
  renameBtn.className = "secondary";
  renameBtn.textContent = "Rename";
  renameBtn.onclick = async () => {
    const title = window.prompt("Session title", info.title || "") || info.title || "";
    const result = await api(`${currentApiBase()}/sessions/${state.selectedSessionId}`, {
      method: "PATCH",
      body: JSON.stringify({ title }),
    });
    if (result.session?.id) {
      await refreshAll();
    }
  };
  sessionActions.appendChild(renameBtn);

  const forkBtn = document.createElement("button");
  forkBtn.className = "secondary";
  forkBtn.textContent = "Fork Last";
  forkBtn.onclick = async () => {
    const messageIndex = Math.max(messages.length - 1, 0);
    const result = await api(`${currentApiBase()}/sessions/${state.selectedSessionId}/fork`, {
      method: "POST",
      body: JSON.stringify({ messageIndex }),
    });
    if (result.session?.id) {
      state.selectedSessionId = result.session.id;
      persistSelection();
      await refreshAll();
    }
  };
  sessionActions.appendChild(forkBtn);

  els.sessionDetail.appendChild(sessionActions);
  if (checkpoints.length) {
    const checkpointActions = document.createElement("div");
    checkpointActions.className = "actions";
    for (const checkpoint of checkpoints.slice(0, 3)) {
      const preview = document.createElement("button");
      preview.className = "secondary";
      preview.textContent = `Preview ${checkpoint.id.slice(0, 6)}`;
      preview.onclick = async () => {
        const meta = await api(`${currentApiBase()}/checkpoints/${checkpoint.id}`);
        const result = await api(`${currentApiBase()}/checkpoints/${checkpoint.id}/diff`);
        bubble(
          "system",
          `${meta.checkpoint?.prompt || checkpoint.prompt || "Checkpoint"}\n\n${result.diff || "No diff available for checkpoint"}`,
        );
      };
      checkpointActions.appendChild(preview);

      const restore = document.createElement("button");
      restore.className = "secondary";
      restore.textContent = `Restore ${checkpoint.id.slice(0, 6)}`;
      restore.onclick = async () => {
        const result = await api(`${currentApiBase()}/sessions/${state.selectedSessionId}/rewind`, {
          method: "POST",
          body: JSON.stringify({ checkpointID: checkpoint.id }),
        });
        bubble(
          "system",
          result.error
            ? `Restore failed: ${result.error}`
            : `Restored checkpoint ${checkpoint.id.slice(0, 8)}\n${(result.modified || []).join("\n")}`,
        );
      };
      checkpointActions.appendChild(restore);
    }
    els.sessionDetail.appendChild(checkpointActions);
  }
  els.sessionDetail.appendChild(
    detailBlock(
      "Todos",
      todos.length
        ? todos.map((todo) => `- [${todo.status}] ${todo.content}`).join("\n")
        : "No todos",
    ),
  );
  els.sessionDetail.appendChild(
    detailBlock(
      "Recent Checkpoints",
      checkpoints.length
        ? checkpoints.slice(0, 5).map((item) => `${item.id.slice(0, 8)} · ${item.file_count} files`).join("\n")
        : "No checkpoints",
    ),
  );
  els.sessionDetail.appendChild(
    detailBlock(
      "Recent Messages",
      messages.slice(-6).map((msg) => `${msg.role}: ${String(msg.content || "").slice(0, 160)}`).join("\n\n") || "No messages",
    ),
  );
  els.sessionDetail.appendChild(
    detailButtonList("Ancestor Sessions", ancestors, async (id) => {
      state.selectedSessionId = id;
      persistSelection();
      await loadSessionDetail();
    }),
  );
  els.sessionDetail.appendChild(
    detailButtonList("Child Sessions", children, async (id) => {
      state.selectedSessionId = id;
      persistSelection();
      await loadSessionDetail();
    }),
  );
}

async function createSession() {
  const result = await api(`${currentApiBase()}/sessions`, {
    method: "POST",
    body: JSON.stringify({
      title: `Web session ${new Date().toLocaleTimeString()}`,
      workspaceRoot: state.selectedWorkspaceId ? "" : undefined,
    }),
  });
  state.selectedSessionId = result.session.id;
  persistSelection();
  await refreshAll();
}

async function importSession() {
  const data = els.importJson.value.trim();
  if (!data) return;
  const result = await api(`${currentApiBase()}/sessions/import`, {
    method: "POST",
    body: JSON.stringify({ data }),
  });
  if (result.session?.id) {
    state.selectedSessionId = result.session.id;
    persistSelection();
    els.importJson.value = "";
    await refreshAll();
  }
}

async function submitChat() {
  const message = els.chatInput.value.trim();
  if (!message) return;
  bubble("user", message);
  els.chatInput.value = "";
  if (els.chatStream.checked) {
    await streamChat(message);
  } else {
    const result = await api(`${currentApiBase()}/chat`, {
      method: "POST",
      body: JSON.stringify({
        message,
        sessionID: state.selectedSessionId,
      }),
    });
    await handleChatResult(result);
  }
  await loadSessionDetail();
}

async function streamChat(message) {
  const response = await fetch(`${currentApiBase()}/chat/stream`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      message,
      sessionID: state.selectedSessionId,
    }),
  });
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let assistant = "";
  let pending = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    pending += decoder.decode(value, { stream: true });
    const parts = pending.split("\n\n");
    pending = parts.pop() || "";
    for (const part of parts) {
      const line = part
        .split("\n")
        .find((item) => item.startsWith("data: "));
      if (!line) continue;
      const payload = JSON.parse(line.slice(6));
      if (payload.type === "chunk") {
        assistant += payload.content || "";
      } else if (payload.type === "done") {
        assistant = payload.content || assistant;
      } else if (payload.type === "question" || payload.type === "toolApproval") {
        if (assistant) bubble("assistant", assistant);
        await handleChatResult(payload);
        return;
      } else if (payload.type === "error") {
        bubble("system", payload.error || "Stream error");
        return;
      }
    }
  }
  bubble("assistant", assistant);
}

async function handleChatResult(result) {
  if (result.questionRequired) {
    const question = result.questionRequired.questions[0];
    const answer = window.prompt(question.question, question.options?.[0] || "");
    const next = await api(`${currentApiBase()}/chat/continue`, {
      method: "POST",
      body: JSON.stringify({
        approvalToken: result.approvalToken,
        answers: [answer || ""],
      }),
    });
    return handleChatResult(next);
  }
  if (result.toolApprovalRequired) {
    const allow = window.confirm(`Allow tool ${result.toolApprovalRequired.tool}?`);
    const next = await api(`${currentApiBase()}/chat/continue`, {
      method: "POST",
      body: JSON.stringify({
        approvalToken: result.approvalToken,
        decision: allow ? "run" : "skip",
      }),
    });
    return handleChatResult(next);
  }
  if (result.error) {
    bubble("system", result.error);
    return;
  }
  bubble("assistant", result.content || "");
}

async function refreshAll() {
  persistSelection();
  ensureEventSource();
  await loadFleetSummary();
  await loadPrincipals();
  await loadWorkspaceInfo();
  await loadWorkspaces();
  await loadSessions();
  await loadSessionDetail();
}

async function loadRecentEvents() {
  const result = await api(`${currentApiBase()}/events/recent`);
  clearChildren(els.activityLog);
  for (const event of (result.events || []).slice().reverse()) {
    prependActivity(event);
  }
}

function ensureEventSource() {
  const nextUrl = `${currentApiBase()}/events`;
  if (state.eventSourceUrl === nextUrl && state.eventSource) return;
  if (state.eventSource) {
    state.eventSource.close();
    state.eventSource = null;
  }
  state.eventSourceUrl = nextUrl;
  state.eventSource = new EventSource(nextUrl);
  state.eventSource.onmessage = async (event) => {
    try {
      const payload = JSON.parse(event.data);
      if (!payload || payload.type === "ready" || payload.type === "ping") return;
      prependActivity(payload);
      await refreshAll();
    } catch (_err) {
      // Ignore malformed event payloads.
    }
  };
}

els.workspaceForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = els.workspaceName.value.trim();
  const baseUrl = els.workspaceUrl.value.trim();
  const apiKey = els.workspaceKey.value.trim();
  if (!name || !baseUrl) return;
  await api("/workspaces", {
    method: "POST",
    body: JSON.stringify({ name, type: "remote", baseUrl, apiKey }),
  });
  els.workspaceName.value = "";
  els.workspaceUrl.value = "";
  els.workspaceKey.value = "";
  await refreshAll();
});

els.principalForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const name = els.principalName.value.trim();
  const role = els.principalRole.value.trim() || "admin";
  if (!name) return;
  await api("/principals", {
    method: "POST",
    body: JSON.stringify({ name, role }),
  });
  els.principalName.value = "";
  els.principalRole.value = "";
  await refreshAll();
});

els.refreshWorkspace.addEventListener("click", refreshAll);
els.sessionSearch.addEventListener("input", loadSessions);
els.newSession.addEventListener("click", createSession);
els.importSession.addEventListener("click", importSession);
els.clearActivity.addEventListener("click", () => clearChildren(els.activityLog));
els.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitChat();
});
els.clearChat.addEventListener("click", () => {
  clearChildren(els.chatLog);
});

await refreshAll();
await loadRecentEvents();
