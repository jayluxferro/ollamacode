/**
 * OllamaCode VS Code extension.
 * Requires `ollamacode serve` running (default http://localhost:8000).
 * Commands: Chat, Chat with selection, Apply edits.
 */

import * as http from "http";
import * as https from "https";
import * as vscode from "vscode";

interface ChatResponse {
  content?: string;
  edits?: Array<{ path: string; oldText?: string; newText: string }>;
  error?: string;
  toolApprovalRequired?: { tool: string; arguments: Record<string, unknown> };
  approvalToken?: string;
  plan?: string;
  review?: Record<string, unknown>;
}

function getConfig() {
  const config = vscode.workspace.getConfiguration("ollamacode");
  const baseUrl = (config.get<string>("baseUrl") || "http://localhost:8000").replace(/\/$/, "");
  const apiKey = config.get<string>("apiKey") || "";
  const confirmToolCalls = config.get<boolean>("confirmToolCalls") || false;
  const multiAgent = config.get<boolean>("multiAgent") || false;
  const multiAgentMaxIterations = config.get<number>("multiAgentMaxIterations") || 2;
  const multiAgentRequireReview = config.get<boolean>("multiAgentRequireReview") ?? true;
  const memoryAutoContext = config.get<boolean>("memoryAutoContext") ?? true;
  const memoryKgMaxResults = config.get<number>("memoryKgMaxResults") || 4;
  const memoryRagMaxResults = config.get<number>("memoryRagMaxResults") || 4;
  const memoryRagSnippetChars = config.get<number>("memoryRagSnippetChars") || 220;
  const enableDiagnostics = config.get<boolean>("enableDiagnostics") || false;
  const enableInlineCompletions = config.get<boolean>("enableInlineCompletions") || false;
  return {
    baseUrl,
    apiKey,
    confirmToolCalls,
    multiAgent,
    multiAgentMaxIterations,
    multiAgentRequireReview,
    memoryAutoContext,
    memoryKgMaxResults,
    memoryRagMaxResults,
    memoryRagSnippetChars,
    enableDiagnostics,
    enableInlineCompletions,
  };
}

function getHeaders(apiKey: string): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (apiKey) {
    headers["Authorization"] = `Bearer ${apiKey}`;
    headers["X-API-Key"] = apiKey;
  }
  return headers;
}

function getWorkspaceRelativePath(uri: vscode.Uri): string | undefined {
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (!folder) return undefined;
  let rel = vscode.workspace.asRelativePath(uri, false);
  if (process.platform === "win32") rel = rel.replace(/\\/g, "/");
  return rel;
}

function httpPost(url: string, body: string, headers: Record<string, string>): Promise<string> {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const isHttps = u.protocol === "https:";
    const opts: https.RequestOptions = {
      hostname: u.hostname,
      port: u.port || (isHttps ? 443 : 80),
      path: u.pathname + u.search,
      method: "POST",
      headers: { ...headers, "Content-Length": Buffer.byteLength(body, "utf8") },
    };
    const req = (isHttps ? https : http).request(opts, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk) => chunks.push(chunk));
      res.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    });
    req.on("error", reject);
    req.write(body, "utf8");
    req.end();
  });
}

async function httpPostJson(
  url: string,
  body: Record<string, unknown>,
  headers: Record<string, string>
): Promise<ChatResponse> {
  const raw = await httpPost(url, JSON.stringify(body), headers);
  return JSON.parse(raw) as ChatResponse;
}

async function requestWithToolApproval(
  baseUrl: string,
  apiKey: string,
  body: Record<string, unknown>,
  onStatus?: (text: string) => void
): Promise<ChatResponse> {
  const headers = getHeaders(apiKey);
  let res = await httpPostJson(`${baseUrl}/chat`, body, headers);
  while (res.toolApprovalRequired && res.approvalToken) {
    const tool = res.toolApprovalRequired.tool;
    const args = res.toolApprovalRequired.arguments;
    if (onStatus) onStatus(`Awaiting approval: ${tool}`);
    const choice = await vscode.window.showQuickPick(
      ["Run", "Skip", "Edit"],
      { placeHolder: `Approve tool call: ${tool}` }
    );
    if (!choice) return { error: "Tool approval cancelled." };
    let decision: "run" | "skip" | "edit" = "run";
    let editedArguments: Record<string, unknown> | undefined;
    if (choice === "Skip") decision = "skip";
    if (choice === "Edit") {
      decision = "edit";
      const edited = await vscode.window.showInputBox({
        prompt: "Edit tool arguments as JSON",
        value: JSON.stringify(args, null, 2),
      });
      if (!edited) return { error: "Tool approval cancelled." };
      try {
        editedArguments = JSON.parse(edited);
      } catch {
        return { error: "Invalid JSON for editedArguments." };
      }
    }
    res = await httpPostJson(
      `${baseUrl}/chat/continue`,
      { approvalToken: res.approvalToken, decision, editedArguments },
      headers
    );
  }
  return res;
}

interface SSEEvent {
  type: "chunk" | "done" | "error";
  content?: string;
  edits?: Array<{ path: string; oldText?: string; newText: string }>;
  error?: string;
}

function httpPostStream(
  url: string,
  body: string,
  headers: Record<string, string>,
  onEvent: (event: SSEEvent) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const u = new URL(url);
    const isHttps = u.protocol === "https:";
    const opts: https.RequestOptions = {
      hostname: u.hostname,
      port: u.port || (isHttps ? 443 : 80),
      path: u.pathname + u.search,
      method: "POST",
      headers: { ...headers, "Content-Length": Buffer.byteLength(body, "utf8") },
    };
    const req = (isHttps ? https : http).request(opts, (res) => {
      let buffer = "";
      res.on("data", (chunk: Buffer) => {
        buffer += chunk.toString("utf8");
        const parts = buffer.split("\n\n");
        buffer = parts.pop() ?? "";
        for (const part of parts) {
          const line = part.split("\n").find((l) => l.startsWith("data: "));
          if (line) {
            try {
              const payload = JSON.parse(line.slice(6)) as SSEEvent;
              onEvent(payload);
              if (payload.type === "done" || payload.type === "error") resolve();
            } catch {
              // skip malformed
            }
          }
        }
      });
      res.on("end", () => resolve());
      res.on("error", reject);
    });
    req.on("error", reject);
    req.write(body, "utf8");
    req.end();
  });
}

export function activate(context: vscode.ExtensionContext) {
  const output = vscode.window.createOutputChannel("OllamaCode");
  const diagnostics = vscode.languages.createDiagnosticCollection("ollamacode");
  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left);
  statusBar.command = "ollamacode.openChatPanel";
  statusBar.text = "OllamaCode: Idle";
  statusBar.show();
  context.subscriptions.push(statusBar);

  let chatPanel: vscode.WebviewPanel | null = null;
  let chatSidebar: vscode.WebviewView | null = null;
  let lastEdits: Array<{ path: string; oldText?: string; newText: string }> = [];

  const setStatus = (text: string) => {
    statusBar.text = `OllamaCode: ${text}`;
  };

  const sendPanelMessage = (payload: Record<string, unknown>) => {
    if (chatPanel) {
      chatPanel.webview.postMessage(payload);
    }
    if (chatSidebar) {
      chatSidebar.webview.postMessage(payload);
    }
  };

  const getChatPanelHtml = () => {
    return `<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      body { font-family: var(--vscode-font-family); margin: 0; padding: 12px; }
      #history { border: 1px solid var(--vscode-editorWidget-border); padding: 8px; height: 55vh; overflow-y: auto; }
      #controls { margin-top: 8px; display: flex; gap: 8px; }
      #msg { flex: 1; }
      .row { margin-bottom: 8px; }
      .toolbar { display: flex; gap: 8px; margin-top: 8px; }
      .label { color: var(--vscode-descriptionForeground); font-size: 0.9em; }
      .row-toolbar { display: flex; gap: 6px; margin-top: 6px; }
      .row-toolbar button { font-size: 0.85em; }
      pre { background: var(--vscode-editor-background); padding: 8px; overflow-x: auto; }
      code { font-family: var(--vscode-editor-font-family); }
      h1, h2, h3, h4 { margin: 10px 0 6px; }
      ul, ol { margin: 6px 0 6px 18px; }
      p { margin: 6px 0; }
      a { color: var(--vscode-textLink-foreground); }
    </style>
  </head>
  <body>
    <div id="history"></div>
    <div id="controls">
      <input id="msg" type="text" placeholder="Message to OllamaCode" />
      <button id="send">Send</button>
    </div>
    <div style="margin-top:8px;">
      <label><input type="checkbox" id="useSelection" /> Use selection</label>
      <label style="margin-left:12px;"><input type="checkbox" id="stream" /> Stream</label>
    </div>
    <div class="toolbar">
      <button id="preview" disabled>Preview edits</button>
      <button id="apply" disabled>Apply edits</button>
      <button id="copy">Copy last reply</button>
      <button id="ragIndex">Index RAG</button>
      <button id="ragQuery">Query RAG</button>
      <button id="export">Export</button>
      <button id="import">Import</button>
      <button id="clear">Clear</button>
    </div>
    <script>
      const vscode = acquireVsCodeApi();
      const history = document.getElementById("history");
      const msg = document.getElementById("msg");
      const send = document.getElementById("send");
      const previewBtn = document.getElementById("preview");
      const applyBtn = document.getElementById("apply");
      const copyBtn = document.getElementById("copy");
      const ragIndexBtn = document.getElementById("ragIndex");
      const ragQueryBtn = document.getElementById("ragQuery");
      const exportBtn = document.getElementById("export");
      const importBtn = document.getElementById("import");
      const clearBtn = document.getElementById("clear");
      let lastReply = "";
      let streamingDiv = null;

      function escapeHtml(s) {
        return s.replace(/[&<>"']/g, (c) => {
          if (c === "&") return "&amp;";
          if (c === "<") return "&lt;";
          if (c === ">") return "&gt;";
          if (c === "\"") return "&quot;";
          return "&#39;";
        });
      }
      function renderInline(text) {
        let t = escapeHtml(text);
        t = t.replace(/\\*\\*(.*?)\\*\\*/g, "<strong>\$1</strong>");
        t = t.replace(/\\*(.*?)\\*/g, "<em>\$1</em>");
        t = t.replace(/\`([^\`]+)\`/g, "<code>\$1</code>");
        t = t.replace(/\\[([^\\]]+)\\]\\((https?:\\/\\/[^)]+)\\)/g, "<a href='\$2'>\$1</a>");
        return t;
      }
      function renderMarkdown(text) {
        const parts = text.split("\`\`\`");
        let html = "";
        for (let i = 0; i < parts.length; i++) {
          if (i % 2 === 1) {
            html += "<pre><code>" + escapeHtml(parts[i]) + "</code></pre>";
            continue;
          }
          const lines = parts[i].split("\\n");
          let inUl = false;
          let inOl = false;
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) {
              if (inUl) { html += "</ul>"; inUl = false; }
              if (inOl) { html += "</ol>"; inOl = false; }
              continue;
            }
            if (trimmed.startsWith("# ")) {
              if (inUl) { html += "</ul>"; inUl = false; }
              if (inOl) { html += "</ol>"; inOl = false; }
              html += "<h1>" + renderInline(trimmed.slice(2)) + "</h1>";
              continue;
            }
            if (trimmed.startsWith("## ")) {
              if (inUl) { html += "</ul>"; inUl = false; }
              if (inOl) { html += "</ol>"; inOl = false; }
              html += "<h2>" + renderInline(trimmed.slice(3)) + "</h2>";
              continue;
            }
            if (/^[-*] /.test(trimmed)) {
              if (inOl) { html += "</ol>"; inOl = false; }
              if (!inUl) { html += "<ul>"; inUl = true; }
              html += "<li>" + renderInline(trimmed.slice(2)) + "</li>";
              continue;
            }
            if (/^\\d+\\. /.test(trimmed)) {
              if (inUl) { html += "</ul>"; inUl = false; }
              if (!inOl) { html += "<ol>"; inOl = true; }
              const afterDot = trimmed.replace(/^\\d+\\.\\s+/, "");
              html += "<li>" + renderInline(afterDot) + "</li>";
              continue;
            }
            if (inUl) { html += "</ul>"; inUl = false; }
            if (inOl) { html += "</ol>"; inOl = false; }
            html += "<p>" + renderInline(trimmed) + "</p>";
          }
          if (inUl) html += "</ul>";
          if (inOl) html += "</ol>";
          if (i < parts.length - 1) {
            html += "";
          }
        }
        return html;
      }
      function append(label, text, opts = {}) {
        const edits = Array.isArray(opts.edits) ? opts.edits : [];
        const isAssistant = label === "Assistant";
        const isUser = label === "You";
        const div = document.createElement("div");
        div.className = "row";
        const bodyHtml = "<div class='label'>" + label + "</div><div>" + renderMarkdown(text) + "</div>";
        let toolbar = "";
        if (isAssistant) {
          toolbar = "<div class='row-toolbar'>"
            + "<button data-action='copy'>Copy</button>"
            + (edits.length > 0 ? "<button data-action='preview'>Preview edits</button><button data-action='apply'>Apply edits</button>" : "")
            + "</div>";
        }
        if (isUser) {
          toolbar = "<div class='row-toolbar'><button data-action='retry'>Retry</button></div>";
        }
        div.innerHTML = bodyHtml + toolbar;
        div.setAttribute("data-text", text);
        div.setAttribute("data-edits", JSON.stringify(edits));
        history.appendChild(div);
        history.scrollTop = history.scrollHeight;
      }
      function sendChat(text, persist = true) {
        if (!text) return;
        append("You", text);
        if (persist) {
          const state = vscode.getState() || { history: [] };
          state.history.push({ role: "user", text });
          vscode.setState(state);
        }
        vscode.postMessage({
          type: "chat",
          message: text,
          useSelection: document.getElementById("useSelection").checked,
          stream: document.getElementById("stream").checked,
        });
      }
      function ensureStreamingRow() {
        if (streamingDiv) return streamingDiv;
        const div = document.createElement("div");
        div.className = "row";
        div.innerHTML = "<div class='label'>Assistant</div><div></div>";
        history.appendChild(div);
        history.scrollTop = history.scrollHeight;
        streamingDiv = div;
        return div;
      }
      function appendStreamingChunk(text) {
        const row = ensureStreamingRow();
        const content = row.querySelector("div:last-child");
        if (!content) return;
        const existing = content.getAttribute("data-raw") || "";
        const next = existing + text;
        content.setAttribute("data-raw", next);
        content.innerHTML = renderMarkdown(next);
        history.scrollTop = history.scrollHeight;
      }
      function finalizeStreaming(fullText, edits = []) {
        if (streamingDiv) {
          streamingDiv.remove();
          streamingDiv = null;
        }
        append("Assistant", fullText, { edits });
        lastReply = fullText;
        const state = vscode.getState() || { history: [] };
        state.history.push({ role: "assistant", text: fullText, edits });
        vscode.setState(state);
      }
      send.addEventListener("click", () => {
        const text = msg.value.trim();
        if (!text) return;
        msg.value = "";
        sendChat(text, true);
      });
      msg.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          send.click();
        }
      });
      previewBtn.addEventListener("click", () => vscode.postMessage({ type: "previewEdits" }));
      applyBtn.addEventListener("click", () => vscode.postMessage({ type: "applyEdits" }));
      copyBtn.addEventListener("click", async () => {
        if (!lastReply) return;
        vscode.postMessage({ type: "copyLast", text: lastReply });
      });
      ragIndexBtn.addEventListener("click", () => vscode.postMessage({ type: "ragIndex" }));
      ragQueryBtn.addEventListener("click", () => vscode.postMessage({ type: "ragQuery" }));
      exportBtn.addEventListener("click", () => {
        const state = vscode.getState() || { history: [] };
        vscode.postMessage({ type: "exportHistory", history: Array.isArray(state.history) ? state.history : [] });
      });
      importBtn.addEventListener("click", () => vscode.postMessage({ type: "importHistory" }));
      clearBtn.addEventListener("click", () => {
        history.innerHTML = "";
        streamingDiv = null;
        const state = { history: [] };
        vscode.setState(state);
        vscode.postMessage({ type: "clearHistory" });
      });
      history.addEventListener("click", (event) => {
        const target = event.target;
        if (!target || target.tagName !== "BUTTON") return;
        const action = target.getAttribute("data-action");
        if (!action) return;
        const row = target.closest(".row");
        if (!row) return;
        const text = row.getAttribute("data-text") || "";
        let edits = [];
        try {
          edits = JSON.parse(row.getAttribute("data-edits") || "[]");
        } catch {
          edits = [];
        }
        if (action === "retry") {
          sendChat(text, true);
          return;
        }
        vscode.postMessage({ type: "messageAction", action, text, edits });
      });

      const state = vscode.getState();
      if (state && Array.isArray(state.history)) {
        for (const item of state.history) {
          append(item.role === "user" ? "You" : "Assistant", item.text, { edits: item.edits || [] });
        }
      }

      window.addEventListener("message", (event) => {
        const msg = event.data;
        if (msg.type === "assistant") {
          if (streamingDiv) {
            streamingDiv.remove();
            streamingDiv = null;
          }
          const edits = Array.isArray(msg.edits) ? msg.edits : [];
          append("Assistant", msg.text, { edits });
          lastReply = msg.text;
          const state = vscode.getState() || { history: [] };
          state.history.push({ role: "assistant", text: msg.text, edits });
          vscode.setState(state);
        }
        if (msg.type === "assistantChunk") appendStreamingChunk(msg.text);
        if (msg.type === "assistantChunkDone" && msg.text) finalizeStreaming(msg.text, Array.isArray(msg.edits) ? msg.edits : []);
        if (msg.type === "importHistory") {
          const imported = Array.isArray(msg.history) ? msg.history : [];
          history.innerHTML = "";
          streamingDiv = null;
          for (const item of imported) {
            const role = item && item.role === "user" ? "You" : "Assistant";
            const text = item && typeof item.text === "string" ? item.text : "";
            const edits = item && Array.isArray(item.edits) ? item.edits : [];
            append(role, text, { edits });
          }
          const state = { history: imported };
          vscode.setState(state);
          lastReply = "";
          for (let i = imported.length - 1; i >= 0; i--) {
            const it = imported[i];
            if (it && it.role === "assistant" && typeof it.text === "string") {
              lastReply = it.text;
              break;
            }
          }
        }
        if (msg.type === "status") append("Status", msg.text);
        if (msg.type === "edits") {
          previewBtn.disabled = msg.count === 0;
          applyBtn.disabled = msg.count === 0;
        }
      });
    ` + "</script>\n  </body>\n</html>";
  };

  async function chat(includeSelection: boolean): Promise<void> {
    const {
      baseUrl,
      apiKey,
      confirmToolCalls,
      multiAgent,
      multiAgentMaxIterations,
      multiAgentRequireReview,
      memoryAutoContext,
      memoryKgMaxResults,
      memoryRagMaxResults,
      memoryRagSnippetChars,
    } = getConfig();
    let message = await vscode.window.showInputBox({
      prompt: "Message to OllamaCode",
      placeHolder: "e.g. Explain this, fix the bug, add tests",
    });
    if (message === undefined) return;

    const editor = vscode.window.activeTextEditor;
    let file: string | undefined;
    let lines: string | undefined;
    if (includeSelection && editor) {
      const uri = editor.document.uri;
      file = getWorkspaceRelativePath(uri);
      const sel = editor.selection;
      if (sel && !sel.isEmpty) {
        lines = `${sel.start.line + 1}-${sel.end.line + 1}`;
      }
    } else if (editor) {
      file = getWorkspaceRelativePath(editor.document.uri);
    }

    const body: Record<string, unknown> = { message, confirmToolCalls, multiAgent };
    body.memoryAutoContext = memoryAutoContext;
    body.memoryKgMaxResults = memoryKgMaxResults;
    body.memoryRagMaxResults = memoryRagMaxResults;
    body.memoryRagSnippetChars = memoryRagSnippetChars;
    if (multiAgent) {
      body.multiAgentMaxIterations = multiAgentMaxIterations;
      body.multiAgentRequireReview = multiAgentRequireReview;
    }
    if (file) body.file = file;
    if (lines) body.lines = lines;

    output.appendLine(`POST ${baseUrl}/chat`);
    try {
      setStatus("Thinking");
      const data = confirmToolCalls
        ? await requestWithToolApproval(baseUrl, apiKey, body, setStatus)
        : await httpPostJson(`${baseUrl}/chat`, body, getHeaders(apiKey));
      setStatus("Idle");
      if (data.error) {
        output.appendLine(`Error: ${data.error}`);
        vscode.window.showErrorMessage(`OllamaCode: ${data.error}`);
        return;
      }
      if (data.content) {
        output.appendLine(data.content);
        output.show();
      }
      if (data.plan) {
        output.appendLine("\n[Plan]\n" + data.plan);
      }
      if (data.review) {
        output.appendLine("\n[Review]\n" + JSON.stringify(data.review, null, 2));
      }
      if (data.edits && data.edits.length > 0) {
        const apply = "Apply edits";
        const preview = "Preview diff";
        const choice = await vscode.window.showInformationMessage(
          `OllamaCode returned ${data.edits.length} edit(s).`,
          preview,
          apply,
          "Dismiss"
        );
        if (choice === preview) {
          await previewEdits(data.edits);
        }
        if (choice === apply) {
          await applyEditsToWorkspace(data.edits);
        }
        lastEdits = data.edits;
        sendPanelMessage({ type: "edits", count: data.edits.length });
      }
    } catch (e) {
      setStatus("Idle");
      const err = e instanceof Error ? e.message : String(e);
      output.appendLine(`Request failed: ${err}`);
      vscode.window.showErrorMessage(
        `OllamaCode: request failed. Is \`ollamacode serve\` running at ${baseUrl}?`
      );
    }
  }

  async function chatStream(includeSelection: boolean): Promise<void> {
    const {
      baseUrl,
      apiKey,
      multiAgent,
      memoryAutoContext,
      memoryKgMaxResults,
      memoryRagMaxResults,
      memoryRagSnippetChars,
    } = getConfig();
    const message = await vscode.window.showInputBox({
      prompt: "Message to OllamaCode (streaming)",
      placeHolder: "e.g. Explain this, fix the bug, add tests",
    });
    if (message === undefined) return;

    const editor = vscode.window.activeTextEditor;
    let file: string | undefined;
    let lines: string | undefined;
    if (includeSelection && editor) {
      const uri = editor.document.uri;
      file = getWorkspaceRelativePath(uri);
      const sel = editor.selection;
      if (sel && !sel.isEmpty) {
        lines = `${sel.start.line + 1}-${sel.end.line + 1}`;
      }
    } else if (editor) {
      file = getWorkspaceRelativePath(editor.document.uri);
    }

    const body: Record<string, unknown> = { message, multiAgent };
    body.memoryAutoContext = memoryAutoContext;
    body.memoryKgMaxResults = memoryKgMaxResults;
    body.memoryRagMaxResults = memoryRagMaxResults;
    body.memoryRagSnippetChars = memoryRagSnippetChars;
    if (file) body.file = file;
    if (lines) body.lines = lines;

    output.clear();
    output.appendLine(`POST ${baseUrl}/chat/stream`);
    output.show();
    try {
      setStatus("Streaming");
      await httpPostStream(
        `${baseUrl}/chat/stream`,
        JSON.stringify(body),
        getHeaders(apiKey),
        (event) => {
          if (event.type === "chunk" && event.content) {
            output.append(event.content);
          }
          if (event.type === "done") {
            if (event.content) output.appendLine("\n");
            if (event.edits && event.edits.length > 0) {
              vscode.window.showInformationMessage(
                `OllamaCode returned ${event.edits.length} edit(s).`,
                "Preview diff",
                "Apply edits",
                "Dismiss"
              ).then((choice) => {
                if (choice === "Preview diff") {
                  previewEdits(event.edits!);
                }
                if (choice === "Apply edits") {
                  applyEditsToWorkspace(event.edits!);
                }
              });
              lastEdits = event.edits!;
              sendPanelMessage({ type: "edits", count: event.edits.length });
            }
            setStatus("Idle");
          }
          if (event.type === "error" && event.error) {
            output.appendLine(`\nError: ${event.error}`);
            vscode.window.showErrorMessage(`OllamaCode: ${event.error}`);
            setStatus("Idle");
          }
        }
      );
    } catch (e) {
      setStatus("Idle");
      const err = e instanceof Error ? e.message : String(e);
      output.appendLine(`\nRequest failed: ${err}`);
      vscode.window.showErrorMessage(
        `OllamaCode: request failed. Is \`ollamacode serve\` running at ${baseUrl}?`
      );
    }
  }

  async function runRagIndex(): Promise<void> {
    const { baseUrl, apiKey } = getConfig();
    const folder = vscode.workspace.workspaceFolders?.[0];
    const suggested = folder?.uri.fsPath || "";
    const workspaceRoot = await vscode.window.showInputBox({
      prompt: "Workspace root to index (leave empty for server default)",
      value: suggested,
      placeHolder: "/path/to/workspace",
    });
    if (workspaceRoot === undefined) return;
    const maxFilesRaw = await vscode.window.showInputBox({
      prompt: "Max files to index (optional)",
      value: "400",
    });
    if (maxFilesRaw === undefined) return;
    const body: Record<string, unknown> = {};
    if (workspaceRoot.trim()) body.workspaceRoot = workspaceRoot.trim();
    const maxFiles = Number(maxFilesRaw);
    if (!Number.isNaN(maxFiles) && maxFiles > 0) body.maxFiles = Math.floor(maxFiles);

    setStatus("Indexing RAG");
    output.appendLine(`POST ${baseUrl}/rag/index`);
    try {
      const data = await httpPostJson(`${baseUrl}/rag/index`, body, getHeaders(apiKey)) as any;
      if (data.error) {
        vscode.window.showErrorMessage(`OllamaCode RAG index failed: ${data.error}`);
        output.appendLine(`RAG index error: ${data.error}`);
      } else {
        const msg = `RAG index built: files=${data.indexed_files ?? "?"}, chunks=${data.chunk_count ?? "?"}`;
        vscode.window.showInformationMessage(msg);
        output.appendLine(msg);
      }
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      output.appendLine(`RAG index request failed: ${err}`);
      vscode.window.showErrorMessage(`OllamaCode RAG index request failed: ${err}`);
    } finally {
      setStatus("Idle");
      output.show();
    }
  }

  async function runRagQuery(): Promise<void> {
    const { baseUrl, apiKey } = getConfig();
    const query = await vscode.window.showInputBox({
      prompt: "RAG query",
      placeHolder: "e.g. auth token refresh flow",
    });
    if (query === undefined) return;
    const q = query.trim();
    if (!q) return;
    setStatus("Querying RAG");
    output.appendLine(`POST ${baseUrl}/rag/query`);
    try {
      const data = await httpPostJson(
        `${baseUrl}/rag/query`,
        { query: q, maxResults: 8 },
        getHeaders(apiKey)
      ) as any;
      if (data.error) {
        vscode.window.showErrorMessage(`OllamaCode RAG query failed: ${data.error}`);
        output.appendLine(`RAG query error: ${data.error}`);
      } else {
        const rows = Array.isArray(data.results) ? data.results : [];
        output.appendLine(`RAG results for: ${q}`);
        if (rows.length === 0) {
          output.appendLine("  (no matches)");
        } else {
          for (const [i, r] of rows.entries()) {
            output.appendLine(`  ${i + 1}. ${r.path ?? "?"} (score=${r.score ?? "?"})`);
            const snippet = String(r.snippet ?? "").replace(/\s+/g, " ").trim();
            if (snippet) output.appendLine(`     ${snippet.slice(0, 220)}${snippet.length > 220 ? "..." : ""}`);
          }
        }
        output.show();
      }
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      output.appendLine(`RAG query request failed: ${err}`);
      vscode.window.showErrorMessage(`OllamaCode RAG query request failed: ${err}`);
    } finally {
      setStatus("Idle");
    }
  }

  async function handleChatWebviewMessage(msg: Record<string, unknown>): Promise<void> {
    if (msg.type === "applyEdits") {
      if (lastEdits.length === 0) {
        sendPanelMessage({ type: "status", text: "No edits to apply." });
        return;
      }
      await applyEditsToWorkspace(lastEdits);
      return;
    }
    if (msg.type === "previewEdits") {
      if (lastEdits.length === 0) {
        sendPanelMessage({ type: "status", text: "No edits to preview." });
        return;
      }
      await previewEdits(lastEdits);
      return;
    }
    if (msg.type === "copyLast") {
      if (typeof msg.text === "string") {
        await vscode.env.clipboard.writeText(msg.text);
        sendPanelMessage({ type: "status", text: "Copied last reply to clipboard." });
      }
      return;
    }
    if (msg.type === "messageAction") {
      const action = typeof msg.action === "string" ? msg.action : "";
      const text = typeof msg.text === "string" ? msg.text : "";
      const edits = Array.isArray(msg.edits)
        ? (msg.edits as Array<{ path: string; oldText?: string; newText: string }>)
        : [];
      if (action === "copy") {
        if (text) {
          await vscode.env.clipboard.writeText(text);
          sendPanelMessage({ type: "status", text: "Copied reply to clipboard." });
        }
        return;
      }
      if (action === "preview") {
        if (edits.length === 0) {
          sendPanelMessage({ type: "status", text: "No edits attached to this reply." });
          return;
        }
        await previewEdits(edits);
        return;
      }
      if (action === "apply") {
        if (edits.length === 0) {
          sendPanelMessage({ type: "status", text: "No edits attached to this reply." });
          return;
        }
        await applyEditsToWorkspace(edits);
        return;
      }
      return;
    }
    if (msg.type === "exportHistory") {
      const raw = Array.isArray(msg.history) ? msg.history : [];
      const historyRows = raw
        .filter((item): item is { role: string; text: string; edits?: Array<{ path: string; oldText?: string; newText: string }> } => {
          return !!item && typeof item === "object" && typeof (item as { role?: unknown }).role === "string";
        })
        .map((item) => ({
          role: item.role === "assistant" ? "assistant" : "user",
          text: typeof item.text === "string" ? item.text : "",
          edits: Array.isArray(item.edits) ? item.edits : [],
        }));

      const target = await vscode.window.showSaveDialog({
        saveLabel: "Export Chat History",
        filters: {
          "JSON": ["json"],
          "Markdown": ["md"],
        },
        defaultUri: vscode.Uri.file("ollamacode-chat-history.json"),
      });
      if (!target) return;

      const asMarkdown = target.path.toLowerCase().endsWith(".md");
      const content = asMarkdown
        ? historyRows.map((row) => {
            const title = row.role === "assistant" ? "Assistant" : "User";
            return `## ${title}\n\n${row.text}\n`;
          }).join("\n")
        : JSON.stringify({ history: historyRows }, null, 2);

      await vscode.workspace.fs.writeFile(target, new TextEncoder().encode(content));
      sendPanelMessage({ type: "status", text: `Exported chat history to ${target.fsPath}` });
      return;
    }
    if (msg.type === "importHistory") {
      const pick = await vscode.window.showOpenDialog({
        canSelectFiles: true,
        canSelectMany: false,
        openLabel: "Import Chat History",
        filters: {
          "JSON": ["json"],
        },
      });
      if (!pick || pick.length === 0) return;
      const uri = pick[0];
      const bytes = await vscode.workspace.fs.readFile(uri);
      const raw = new TextDecoder().decode(bytes);
      try {
        const parsed = JSON.parse(raw) as { history?: Array<{ role?: string; text?: string; edits?: Array<{ path: string; oldText?: string; newText: string }> }> };
        const imported = Array.isArray(parsed.history)
          ? parsed.history.map((item) => ({
              role: item.role === "assistant" ? "assistant" : "user",
              text: typeof item.text === "string" ? item.text : "",
              edits: Array.isArray(item.edits) ? item.edits : [],
            }))
          : [];
        sendPanelMessage({ type: "importHistory", history: imported });
        sendPanelMessage({ type: "status", text: `Imported ${imported.length} chat entries.` });
      } catch {
        sendPanelMessage({ type: "status", text: "Failed to import JSON history." });
      }
      return;
    }
    if (msg.type === "clearHistory") {
      sendPanelMessage({ type: "status", text: "Chat history cleared in this view." });
      return;
    }
    if (msg.type === "ragIndex") {
      await runRagIndex();
      return;
    }
    if (msg.type === "ragQuery") {
      await runRagQuery();
      return;
    }
    if (msg.type !== "chat") return;
    const {
      baseUrl,
      apiKey,
      confirmToolCalls,
      multiAgent,
      multiAgentMaxIterations,
      multiAgentRequireReview,
      memoryAutoContext,
      memoryKgMaxResults,
      memoryRagMaxResults,
      memoryRagSnippetChars,
    } = getConfig();
    const includeSelection = !!(msg.useSelection as boolean);
    const stream = !!(msg.stream as boolean);
    const message = String(msg.message ?? "");
    const editor = vscode.window.activeTextEditor;
    let file: string | undefined;
    let lines: string | undefined;
    if (includeSelection && editor) {
      const uri = editor.document.uri;
      file = getWorkspaceRelativePath(uri);
      const sel = editor.selection;
      if (sel && !sel.isEmpty) {
        lines = `${sel.start.line + 1}-${sel.end.line + 1}`;
      }
    } else if (editor) {
      file = getWorkspaceRelativePath(editor.document.uri);
    }
    const body: Record<string, unknown> = { message, confirmToolCalls, multiAgent };
    body.memoryAutoContext = memoryAutoContext;
    body.memoryKgMaxResults = memoryKgMaxResults;
    body.memoryRagMaxResults = memoryRagMaxResults;
    body.memoryRagSnippetChars = memoryRagSnippetChars;
    if (multiAgent) {
      body.multiAgentMaxIterations = multiAgentMaxIterations;
      body.multiAgentRequireReview = multiAgentRequireReview;
    }
    if (file) body.file = file;
    if (lines) body.lines = lines;
    setStatus(stream ? "Streaming" : "Thinking");
    sendPanelMessage({ type: "status", text: "Sending request..." });
    try {
      if (stream) {
        await httpPostStream(
          `${baseUrl}/chat/stream`,
          JSON.stringify(body),
          getHeaders(apiKey),
          (event) => {
            if (event.type === "chunk" && event.content) {
              sendPanelMessage({ type: "assistantChunk", text: event.content });
            }
            if (event.type === "done") {
              if (event.content) {
                sendPanelMessage({ type: "assistantChunkDone", text: event.content, edits: event.edits ?? [] });
              }
              if (event.edits && event.edits.length > 0) {
                sendPanelMessage({ type: "status", text: `Edits returned: ${event.edits.length}. Use command: Apply edits.` });
                sendPanelMessage({ type: "edits", count: event.edits.length });
                lastEdits = event.edits;
              }
              setStatus("Idle");
            }
            if (event.type === "error" && event.error) {
              sendPanelMessage({ type: "status", text: `Error: ${event.error}` });
              setStatus("Idle");
            }
          }
        );
      } else {
        const data = confirmToolCalls
          ? await requestWithToolApproval(baseUrl, apiKey, body, setStatus)
          : await httpPostJson(`${baseUrl}/chat`, body, getHeaders(apiKey));
        if (data.error) {
          sendPanelMessage({ type: "status", text: `Error: ${data.error}` });
          setStatus("Idle");
          return;
        }
        if (data.content) sendPanelMessage({ type: "assistant", text: data.content, edits: data.edits ?? [] });
        if (data.plan) sendPanelMessage({ type: "status", text: `[Plan] ${data.plan}` });
        if (data.review) sendPanelMessage({ type: "status", text: `[Review] ${JSON.stringify(data.review)}` });
        if (data.edits && data.edits.length > 0) {
          sendPanelMessage({ type: "status", text: `Edits returned: ${data.edits.length}.` });
          sendPanelMessage({ type: "edits", count: data.edits.length });
          lastEdits = data.edits;
        }
        setStatus("Idle");
      }
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      sendPanelMessage({ type: "status", text: `Request failed: ${err}` });
      setStatus("Idle");
    }
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("ollamacode.focusChatView", () => {
      void vscode.commands.executeCommand("workbench.view.extension.ollamacode");
    }),
    vscode.commands.registerCommand("ollamacode.chat", () => chat(false)),
    vscode.commands.registerCommand("ollamacode.chatWithSelection", () => chat(true)),
    vscode.commands.registerCommand("ollamacode.chatStream", () => chatStream(false)),
    vscode.commands.registerCommand("ollamacode.chatStreamWithSelection", () => chatStream(true)),
    vscode.commands.registerCommand("ollamacode.openChatPanel", () => {
      if (chatPanel) {
        chatPanel.reveal();
        return;
      }
      chatPanel = vscode.window.createWebviewPanel(
        "ollamacodeChat",
        "OllamaCode Chat",
        vscode.ViewColumn.Beside,
        { enableScripts: true }
      );
      chatPanel.webview.html = getChatPanelHtml();
      chatPanel.onDidDispose(() => {
        chatPanel = null;
      });
      chatPanel.webview.onDidReceiveMessage((msg) => handleChatWebviewMessage(msg as Record<string, unknown>));
    }),
    vscode.window.registerWebviewViewProvider(
      "ollamacode.chatView",
      {
        resolveWebviewView(
          webviewView: vscode.WebviewView,
          _context: vscode.WebviewViewResolveContext,
          _token: vscode.CancellationToken
        ) {
          chatSidebar = webviewView;
          webviewView.webview.options = { enableScripts: true };
          webviewView.webview.html = getChatPanelHtml();
          webviewView.webview.onDidReceiveMessage((msg) => handleChatWebviewMessage(msg as Record<string, unknown>));
          webviewView.onDidDispose(() => {
            chatSidebar = null;
          });
        },
      },
      { webviewOptions: { retainContextWhenHidden: true } }
    ),
    vscode.commands.registerCommand("ollamacode.runDiagnostics", async () => {
      const { baseUrl, apiKey } = getConfig();
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showErrorMessage("OllamaCode: open a file to run diagnostics.");
        return;
      }
      const file = getWorkspaceRelativePath(editor.document.uri);
      if (!file) {
        vscode.window.showErrorMessage("OllamaCode: file must be inside workspace.");
        return;
      }
      const data = await httpPostJson(
        `${baseUrl}/diagnostics`,
        { path: file },
        getHeaders(apiKey)
      );
      const diags = (data as any).diagnostics || [];
      const vscodeDiags = diags.map((d: any) => {
        const range = new vscode.Range(
          new vscode.Position(d.range.start.line, d.range.start.character),
          new vscode.Position(d.range.end.line, d.range.end.character)
        );
        const sev = d.severity === 1 ? vscode.DiagnosticSeverity.Error :
          d.severity === 2 ? vscode.DiagnosticSeverity.Warning :
          vscode.DiagnosticSeverity.Information;
        return new vscode.Diagnostic(range, d.message, sev);
      });
      diagnostics.set(editor.document.uri, vscodeDiags);
      vscode.window.showInformationMessage(`OllamaCode: diagnostics updated (${vscodeDiags.length}).`);
    }),
    vscode.commands.registerCommand("ollamacode.runRagIndex", () => runRagIndex()),
    vscode.commands.registerCommand("ollamacode.runRagQuery", () => runRagQuery()),
    vscode.commands.registerCommand("ollamacode.applyEdits", async () => {
      const text = await vscode.env.clipboard.readText();
      let edits: Array<{ path: string; oldText?: string; newText: string }>;
      try {
        const parsed = JSON.parse(text);
        edits = Array.isArray(parsed) ? parsed : parsed.edits ? parsed.edits : [];
      } catch {
        vscode.window.showErrorMessage("OllamaCode: clipboard is not valid JSON (expect edits array or { edits: [...] }).");
        return;
      }
      if (edits.length === 0) {
        vscode.window.showWarningMessage("OllamaCode: no edits in clipboard.");
        return;
      }
      await applyEditsToWorkspace(edits);
    }),
    vscode.commands.registerCommand("ollamacode.previewEdits", async () => {
      const text = await vscode.env.clipboard.readText();
      let edits: Array<{ path: string; oldText?: string; newText: string }>;
      try {
        const parsed = JSON.parse(text);
        edits = Array.isArray(parsed) ? parsed : parsed.edits ? parsed.edits : [];
      } catch {
        vscode.window.showErrorMessage("OllamaCode: clipboard is not valid JSON (expect edits array or { edits: [...] }).");
        return;
      }
      if (edits.length === 0) {
        vscode.window.showWarningMessage("OllamaCode: no edits in clipboard.");
        return;
      }
      await previewEdits(edits);
    })
  );

  if (getConfig().enableInlineCompletions) {
    context.subscriptions.push(
      vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, {
        async provideInlineCompletionItems(document, position) {
          const { baseUrl, apiKey } = getConfig();
          const line = document.lineAt(position.line).text.slice(0, position.character);
          if (!line.trim()) return;
          const data = await httpPostJson(
            `${baseUrl}/complete`,
            { prefix: line },
            getHeaders(apiKey)
          );
          const completions = (data as any).completions || [];
          return completions.map((c: string) => ({
            insertText: c,
            range: new vscode.Range(position, position),
          }));
        },
      })
    );
  }
}

async function previewEdits(
  edits: Array<{ path: string; oldText?: string; newText: string }>
): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    vscode.window.showErrorMessage("OllamaCode: open a workspace folder first.");
    return;
  }
  for (const e of edits) {
    const uri = vscode.Uri.joinPath(folder.uri, e.path);
    let originalText = "";
    let language = "plaintext";
    try {
      const doc = await vscode.workspace.openTextDocument(uri);
      originalText = doc.getText();
      language = doc.languageId || language;
    } catch {
      // file may not exist yet
    }
    let updatedText = e.newText;
    if (e.oldText && originalText) {
      const idx = originalText.indexOf(e.oldText);
      if (idx >= 0) {
        updatedText =
          originalText.slice(0, idx) + e.newText + originalText.slice(idx + e.oldText.length);
      }
    } else if (originalText) {
      updatedText = e.newText;
    }
    const originalDoc = await vscode.workspace.openTextDocument({ content: originalText, language });
    const updatedDoc = await vscode.workspace.openTextDocument({ content: updatedText, language });
    const title = `OllamaCode: ${e.path}`;
    await vscode.commands.executeCommand("vscode.diff", originalDoc.uri, updatedDoc.uri, title);
  }
}

async function applyEditsToWorkspace(
  edits: Array<{ path: string; oldText?: string; newText: string }>
): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) {
    vscode.window.showErrorMessage("OllamaCode: open a workspace folder first.");
    return;
  }
  const workspaceEdit = new vscode.WorkspaceEdit();
  for (const e of edits) {
    const uri = vscode.Uri.joinPath(folder.uri, e.path);
    let doc: vscode.TextDocument | null = null;
    try {
      doc = await vscode.workspace.openTextDocument(uri);
    } catch {
      // file may not exist yet
    }
    if (e.oldText !== undefined && e.oldText !== "" && doc) {
      const start = doc.getText().indexOf(e.oldText);
      if (start >= 0) {
        const pos = doc.positionAt(start);
        const end = doc.positionAt(start + e.oldText.length);
        workspaceEdit.replace(uri, new vscode.Range(pos, end), e.newText);
      } else {
        const fullRange = new vscode.Range(0, 0, doc.lineCount, 0);
        workspaceEdit.replace(uri, fullRange, e.newText);
      }
    } else {
      const range = doc
        ? new vscode.Range(0, 0, doc.lineCount, 0)
        : new vscode.Range(0, 0, 0, 0);
      workspaceEdit.replace(uri, range, e.newText);
    }
  }
  const applied = await vscode.workspace.applyEdit(workspaceEdit);
  vscode.window.showInformationMessage(
    applied ? `OllamaCode: applied ${edits.length} edit(s).` : "OllamaCode: failed to apply some edits."
  );
}

export function deactivate() {}
