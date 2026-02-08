import * as vscode from "vscode";
import {
  runCli,
  applyEdits,
  getEditsByFile,
  applyEditsToContent,
  type EditSpec,
} from "./ollamacodeRunner";

const VIEW_TYPE = "ollamacode.chatView";
const COMPOSER_VIEW_TYPE = "ollamacode.composerView";
const CHAT_PARTICIPANT_ID = "ollamacode.chatParticipant";

const OLLAMA_BASE = "http://127.0.0.1:11434";

/** Fetch model names from Ollama API (GET /api/tags). Returns [] if Ollama is unreachable. */
async function fetchOllamaModels(): Promise<string[]> {
  try {
    const res = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(5000) });
    if (!res.ok) return [];
    const data = (await res.json()) as { models?: { name: string }[] };
    const names = data.models?.map((m) => m.name) ?? [];
    return names.length > 0 ? names : [];
  } catch {
    return [];
  }
}

/** If injectCurrentFile is on and there is an active editor in the workspace, prepend current file context to the prompt. */
function promptWithFileContext(prompt: string): string {
  const config = vscode.workspace.getConfiguration("ollamacode");
  if (!config.get<boolean>("injectCurrentFile", true)) return prompt;
  const editor = vscode.window.activeTextEditor;
  if (!editor) return prompt;
  const uri = editor.document.uri;
  const folder = vscode.workspace.getWorkspaceFolder(uri);
  if (!folder) return prompt;
  const relative = vscode.workspace.asRelativePath(uri);
  return `Current file: ${relative}\n\n${prompt}`;
}

/** Open side-by-side diff for each file in edits. */
async function openDiffsForEdits(edits: EditSpec[]): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) return;
  const byFile = getEditsByFile(edits);
  for (const [path, fileEdits] of byFile) {
    const uri = vscode.Uri.joinPath(folder.uri, path);
    const doc = await Promise.resolve(vscode.workspace.openTextDocument(uri)).catch(() => null);
    const current = doc ? doc.getText() : "";
    const newContent = applyEditsToContent(current, fileEdits);
    const lang = doc?.languageId ?? "plaintext";
    const newDoc = await vscode.workspace.openTextDocument({
      content: newContent,
      language: lang,
    });
    await vscode.commands.executeCommand("vscode.diff", uri, newDoc.uri, `OllamaCode: ${path}`);
  }
}

/** Open side-by-side diff for one file (path + its edits). */
async function openDiffForFile(path: string, fileEdits: EditSpec[]): Promise<void> {
  const folder = vscode.workspace.workspaceFolders?.[0];
  if (!folder) return;
  const uri = vscode.Uri.joinPath(folder.uri, path);
  const doc = await Promise.resolve(vscode.workspace.openTextDocument(uri)).catch(() => null);
  const current = doc ? doc.getText() : "";
  const newContent = applyEditsToContent(current, fileEdits);
  const lang = doc?.languageId ?? "plaintext";
  const newDoc = await vscode.workspace.openTextDocument({
    content: newContent,
    language: lang,
  });
  await vscode.commands.executeCommand("vscode.diff", uri, newDoc.uri, `OllamaCode: ${path}`);
}

/** Apply only edits for the given path. */
function applyEditsForPath(edits: EditSpec[], path: string): EditSpec[] {
  return edits.filter((e) => e.path === path);
}

/** If review setting is on, show Apply all / Reject / Show diff; then apply or not. Else apply directly. */
async function applyEditsWithReview(edits: EditSpec[]): Promise<boolean> {
  const config = vscode.workspace.getConfiguration("ollamacode");
  const review = config.get<boolean>("reviewEditsBeforeApply", false);
  if (!review || edits.length === 0) {
    return applyEdits(edits);
  }

  const choice = await vscode.window.showQuickPick(
    [
      { label: "Apply all", value: "apply" as const },
      { label: "Reject", value: "reject" as const },
      { label: "Show diff first", value: "diff" as const },
    ],
    {
      title: `OllamaCode suggested ${edits.length} edit(s)`,
      placeHolder: "Apply all, reject, or show diff first",
    }
  );
  if (!choice || choice.value === "reject") {
    return false;
  }
  if (choice.value === "diff") {
    await openDiffsForEdits(edits);
    const again = await vscode.window.showQuickPick(
      [
        { label: "Apply all", value: "apply" as const },
        { label: "Reject", value: "reject" as const },
      ],
      { title: "Apply these edits?", placeHolder: "Apply all or Reject" }
    );
    if (!again || again.value === "reject") {
      return false;
    }
  }
  return applyEdits(edits);
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(
    vscode.commands.registerCommand("ollamacode.openPanel", () => {
      vscode.commands.executeCommand("workbench.view.extension.ollamacode");
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("ollamacode.selectModel", async () => {
      const models = await fetchOllamaModels();
      if (models.length === 0) {
        void vscode.window.showErrorMessage(
          "OllamaCode: Could not fetch models. Is Ollama running? (http://127.0.0.1:11434)"
        );
        return;
      }
      const config = vscode.workspace.getConfiguration("ollamacode");
      const current = config.get<string>("model", "qwen2.5-coder:32b");
      const picked = await vscode.window.showQuickPick(models, {
        title: "Select Ollama model",
        placeHolder: `Current: ${current}`,
        matchOnDescription: false,
      });
      if (picked) {
        await config.update("model", picked, vscode.ConfigurationTarget.Global);
        void vscode.window.showInformationMessage(`OllamaCode: Using model "${picked}"`);
      }
    })
  );

  const chatProvider = new OllamaCodeChatProvider(context);
  context.subscriptions.push(
    vscode.commands.registerCommand("ollamacode.chatWithSelection", async () => {
      const editor = vscode.window.activeTextEditor;
      const selection = editor?.document.getText(editor.selection)?.trim();
      await vscode.commands.executeCommand("workbench.view.extension.ollamacode");
      if (selection) {
        const query = `Selected code:\n\`\`\`\n${selection}\n\`\`\`\n\nWhat would you like to do with this code?`;
        chatProvider.runQuery(query);
      }
    })
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW_TYPE, chatProvider)
  );

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      COMPOSER_VIEW_TYPE,
      new OllamaCodeComposerProvider(context)
    )
  );

  // Chat Participant: @ollamacode in VS Code Chat (requires VS Code 1.109+)
  if (typeof (vscode.chat as { createChatParticipant?: unknown }).createChatParticipant === "function") {
    const participant = (vscode.chat as { createChatParticipant: (id: string, handler: vscode.ChatRequestHandler) => vscode.ChatParticipant }).createChatParticipant(
      CHAT_PARTICIPANT_ID,
      chatRequestHandler as vscode.ChatRequestHandler
    );
    context.subscriptions.push(participant);
  }
}

const chatRequestHandler: vscode.ChatRequestHandler = async (
  request,
  context,
  stream,
  token
) => {
  const prompt = request.prompt;
  if (!prompt) {
    stream.markdown("No prompt provided.");
    return;
  }

  stream.progress("Running OllamaCode…");
  const config = vscode.workspace.getConfiguration("ollamacode");
  const useStream = config.get<boolean>("streamResponse", true);
  const fullPrompt = promptWithFileContext(prompt);
  const result = await runCli(fullPrompt, {
    injectEditProtocol: true,
    stream: useStream,
    onStreamChunk: useStream ? (chunk) => stream.markdown(chunk) : undefined,
  });

  if (result.code !== 0 && result.stderr) {
    stream.markdown(`Error (exit ${result.code}):\n\n\`\`\`\n${result.stderr}\n\`\`\``);
    return;
  }

  if (result.edits.length > 0) {
    const applied = await applyEditsWithReview(result.edits);
    if (applied) {
      stream.progress(`Applied ${result.edits.length} edit(s) to the workspace.`);
    }
  }

  if (!useStream) {
    const text = result.text || "(no output)";
    stream.markdown(text);
  }
};

class OllamaCodeChatProvider implements vscode.WebviewViewProvider {
  private _view: vscode.WebviewView | undefined;

  constructor(private readonly _context: vscode.ExtensionContext) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this._view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [],
    };
    webviewView.webview.html = getChatHtml(webviewView.webview);

    const history = this._context.workspaceState.get<{ role: string; content: string }[]>("ollamacode.chatHistory") ?? [];
    this.postMessage({ type: "loadHistory", messages: history });

    webviewView.webview.onDidReceiveMessage(
      async (data: { type: string; query?: string; model?: string }) => {
        if (data.type === "send" && data.query) {
          await this.runQuery(data.query);
        }
        if (data.type === "getModels") {
          const models = await fetchOllamaModels();
          const config = vscode.workspace.getConfiguration("ollamacode");
          const current = config.get<string>("model", "qwen2.5-coder:32b");
          this.postMessage({ type: "models", models, current });
        }
        if (data.type === "setModel" && data.model) {
          const config = vscode.workspace.getConfiguration("ollamacode");
          await config.update("model", data.model, vscode.ConfigurationTarget.Global);
          this.postMessage({ type: "modelSet", model: data.model });
        }
      }
    );
  }

  async runQuery(query: string) {
    this.postMessage({ type: "status", text: "Running OllamaCode…" });
    this.postMessage({ type: "append", role: "user", text: query });

    const config = vscode.workspace.getConfiguration("ollamacode");
    const useStream = config.get<boolean>("streamResponse", true);
    const fullPrompt = promptWithFileContext(query);
    this.postMessage({ type: "streamStart" });
    const result = await runCli(fullPrompt, {
      injectEditProtocol: true,
      stream: useStream,
      onStreamChunk: useStream
        ? (chunk) => this.postMessage({ type: "streamChunk", text: chunk })
        : undefined,
    });

    this.postMessage({ type: "streamEnd" });
    this.postMessage({ type: "status", text: "" });

    let assistantText = "";
    if (result.code !== 0 && result.stderr) {
      assistantText = `Error (exit ${result.code}):\n${result.stderr}`;
      this.postMessage({ type: "append", role: "assistant", text: assistantText });
    } else {
      if (!useStream) {
        assistantText = result.text || "(no output)";
        this.postMessage({ type: "append", role: "assistant", text: assistantText });
      } else if (result.edits.length > 0 || result.text) {
        this.postMessage({
          type: "streamResult",
          editsCount: result.edits.length,
          text: result.text || "",
        });
        assistantText = result.text || "";
        if (result.edits.length > 0) {
          const applied = await applyEditsWithReview(result.edits);
          if (applied) {
            assistantText += `\n\n[Applied ${result.edits.length} edit(s) to the workspace.]`;
            this.postMessage({
              type: "append",
              role: "assistant",
              text: `[Applied ${result.edits.length} edit(s) to the workspace.]`,
            });
          }
        }
      }
    }

    const maxHistory = 20;
    const prev = this._context.workspaceState.get<{ role: string; content: string }[]>("ollamacode.chatHistory") ?? [];
    const next = [...prev, { role: "user", content: query }, { role: "assistant", content: assistantText }].slice(-maxHistory * 2);
    this._context.workspaceState.update("ollamacode.chatHistory", next);
  }

  private postMessage(msg: object) {
    this._view?.webview.postMessage(msg);
  }
}

/** Composer: multi-file task → proposed changes list → Preview / Apply / Reject per file or all. */
class OllamaCodeComposerProvider implements vscode.WebviewViewProvider {
  private _view: vscode.WebviewView | undefined;
  private _pendingEdits: EditSpec[] = [];
  private _pendingText = "";

  constructor(private readonly _context: vscode.ExtensionContext) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ) {
    this._view = webviewView;
    webviewView.webview.options = { enableScripts: true, localResourceRoots: [] };
    webviewView.webview.html = getComposerHtml(webviewView.webview);

    webviewView.webview.onDidReceiveMessage(
      async (data: { type: string; prompt?: string; path?: string }) => {
        if (data.type === "run" && data.prompt) {
          await this.runComposer(data.prompt);
        } else if (data.type === "previewFile" && data.path) {
          await this.previewFile(data.path);
        } else if (data.type === "applyFile" && data.path) {
          await this.applyFile(data.path);
        } else if (data.type === "rejectFile" && data.path) {
          this.rejectFile(data.path);
        } else if (data.type === "applyAll") {
          await this.applyAll();
        } else if (data.type === "rejectAll") {
          this.rejectAll();
        }
      }
    );
  }

  private async runComposer(prompt: string) {
    this.postMessage({ type: "status", text: "Running OllamaCode…" });
    this.postMessage({ type: "composerPrompt", text: prompt });

    const fullPrompt = promptWithFileContext(prompt);
    const result = await runCli(fullPrompt, { injectEditProtocol: true });

    this.postMessage({ type: "status", text: "" });

    if (result.code !== 0 && result.stderr) {
      this.postMessage({
        type: "composerResult",
        text: `Error (exit ${result.code}):\n${result.stderr}`,
        files: [],
      });
      return;
    }

    this._pendingEdits = result.edits;
    this._pendingText = result.text || "";
    const byFile = getEditsByFile(result.edits);
    const files = Array.from(byFile.keys());

    this.postMessage({
      type: "composerResult",
      text: this._pendingText,
      files,
    });
  }

  private async previewFile(path: string) {
    const byFile = getEditsByFile(this._pendingEdits);
    const fileEdits = byFile.get(path);
    if (fileEdits) {
      await openDiffForFile(path, fileEdits);
    }
  }

  private async applyFile(path: string) {
    const toApply = applyEditsForPath(this._pendingEdits, path);
    if (toApply.length === 0) return;
    const ok = await applyEdits(toApply);
    if (ok) {
      this._pendingEdits = this._pendingEdits.filter((e) => e.path !== path);
      this.postMessage({ type: "composerUpdate", files: Array.from(getEditsByFile(this._pendingEdits).keys()) });
    }
  }

  private rejectFile(path: string) {
    this._pendingEdits = this._pendingEdits.filter((e) => e.path !== path);
    this.postMessage({ type: "composerUpdate", files: Array.from(getEditsByFile(this._pendingEdits).keys()) });
  }

  private async applyAll() {
    if (this._pendingEdits.length === 0) return;
    const ok = await applyEdits(this._pendingEdits);
    if (ok) {
      this._pendingEdits = [];
      this.postMessage({ type: "composerUpdate", files: [] });
    }
  }

  private rejectAll() {
    this._pendingEdits = [];
    this.postMessage({ type: "composerUpdate", files: [] });
  }

  private postMessage(msg: object) {
    this._view?.webview.postMessage(msg);
  }
}

function getChatHtml(webview: vscode.Webview): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 8px; font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); color: var(--vscode-foreground); }
    #messages { min-height: 120px; max-height: 60vh; overflow-y: auto; margin-bottom: 8px; }
    .msg { margin: 6px 0; padding: 6px 8px; border-radius: 6px; white-space: pre-wrap; word-break: break-word; }
    .msg.user { background: var(--vscode-input-background); }
    .msg.assistant { background: var(--vscode-editor-inactiveSelectionBackground); }
    #status { font-size: 0.9em; color: var(--vscode-descriptionForeground); margin-bottom: 6px; }
    form { display: flex; gap: 6px; }
    input { flex: 1; padding: 6px 8px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); border-radius: 4px; }
    button { padding: 6px 12px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; }
    button:hover { background: var(--vscode-button-hoverBackground); }
  </style>
</head>
<body>
  <div id="status"></div>
  <div id="messages"></div>
  <form id="form">
    <input type="text" id="input" placeholder="Ask OllamaCode…" />
    <button type="submit">Send</button>
  </form>
  <script>
    const vscode = acquireVsCodeApi();
    const statusEl = document.getElementById('status');
    const messagesEl = document.getElementById('messages');
    const formEl = document.getElementById('form');
    const inputEl = document.getElementById('input');

    let streamTarget = null;
    window.addEventListener('message', e => {
      const msg = e.data;
      if (msg.type === 'status') { statusEl.textContent = msg.text || ''; return; }
      if (msg.type === 'streamStart') {
        const div = document.createElement('div');
        div.className = 'msg assistant';
        messagesEl.appendChild(div);
        streamTarget = div;
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return;
      }
      if (msg.type === 'streamChunk' && streamTarget) {
        streamTarget.textContent += msg.text || '';
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return;
      }
      if (msg.type === 'streamEnd') { streamTarget = null; return; }
      if (msg.type === 'streamResult' && streamTarget) {
        if (msg.editsCount > 0) streamTarget.textContent += '\n\n[Applied ' + msg.editsCount + ' edit(s) to the workspace.]';
        streamTarget = null;
        return;
      }
      if (msg.type === 'append') {
        const div = document.createElement('div');
        div.className = 'msg ' + msg.role;
        div.textContent = msg.text;
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      if (msg.type === 'loadHistory' && Array.isArray(msg.messages)) {
        msg.messages.forEach(m => {
          const div = document.createElement('div');
          div.className = 'msg ' + (m.role || 'assistant');
          div.textContent = m.content || '';
          messagesEl.appendChild(div);
        });
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
    });

    formEl.addEventListener('submit', e => {
      e.preventDefault();
      const q = inputEl.value.trim();
      if (!q) return;
      vscode.postMessage({ type: 'send', query: q });
      inputEl.value = '';
    });
  </script>
</body>
</html>`;
}

function getComposerHtml(webview: vscode.Webview): string {
  return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 8px; font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); color: var(--vscode-foreground); }
    #status { font-size: 0.9em; color: var(--vscode-descriptionForeground); margin-bottom: 6px; }
    .section { margin-bottom: 12px; }
    .section-title { font-weight: 600; margin-bottom: 6px; }
    #prompt-row { display: flex; gap: 6px; margin-bottom: 8px; }
    #prompt { flex: 1; padding: 8px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); border-radius: 4px; min-height: 60px; resize: vertical; }
    button { padding: 6px 12px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; white-space: nowrap; }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
    button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
    #response { min-height: 60px; max-height: 40vh; overflow-y: auto; padding: 8px; background: var(--vscode-editor-inactiveSelectionBackground); border-radius: 4px; white-space: pre-wrap; word-break: break-word; margin-bottom: 8px; }
    #files { list-style: none; padding: 0; margin: 0 0 8px 0; }
    #files li { display: flex; align-items: center; gap: 8px; padding: 6px 8px; margin-bottom: 4px; background: var(--vscode-input-background); border-radius: 4px; }
    #files li .path { flex: 1; font-family: var(--vscode-editor-font-family); font-size: 0.9em; overflow: hidden; text-overflow: ellipsis; }
    #files li .actions { display: flex; gap: 4px; }
    #bulk-actions { display: flex; gap: 6px; margin-top: 8px; }
  </style>
</head>
<body>
  <div id="status"></div>
  <div class="section">
    <div class="section-title">Describe your multi-file task</div>
    <div id="prompt-row">
      <textarea id="prompt" placeholder="e.g. Add error handling to all API routes in src/…"></textarea>
      <button id="run-btn" type="button">Run</button>
    </div>
  </div>
  <div id="result-section" style="display: none;">
    <div class="section-title">Response</div>
    <div id="response"></div>
    <div id="changes-section" style="display: none;">
      <div class="section-title">Proposed changes</div>
      <ul id="files"></ul>
      <div id="bulk-actions">
        <button id="apply-all-btn" type="button">Apply all</button>
        <button id="reject-all-btn" type="button" class="secondary">Reject all</button>
      </div>
    </div>
  </div>
  <script>
    const vscode = acquireVsCodeApi();
    const statusEl = document.getElementById('status');
    const promptEl = document.getElementById('prompt');
    const runBtn = document.getElementById('run-btn');
    const resultSection = document.getElementById('result-section');
    const responseEl = document.getElementById('response');
    const changesSection = document.getElementById('changes-section');
    const filesEl = document.getElementById('files');
    const applyAllBtn = document.getElementById('apply-all-btn');
    const rejectAllBtn = document.getElementById('reject-all-btn');

    function renderFiles(files) {
      filesEl.innerHTML = '';
      files.forEach(path => {
        const li = document.createElement('li');
        li.dataset.path = path;
        const pathSpan = document.createElement('span');
        pathSpan.className = 'path';
        pathSpan.title = path;
        pathSpan.textContent = path;
        const actions = document.createElement('span');
        actions.className = 'actions';
        ['Preview', 'Apply', 'Reject'].forEach((label, i) => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.textContent = label;
          if (label === 'Reject') btn.className = 'secondary';
          btn.dataset.action = label.toLowerCase();
          btn.addEventListener('click', () => {
            if (btn.dataset.action === 'preview') vscode.postMessage({ type: 'previewFile', path });
            if (btn.dataset.action === 'apply') vscode.postMessage({ type: 'applyFile', path });
            if (btn.dataset.action === 'reject') vscode.postMessage({ type: 'rejectFile', path });
          });
          actions.appendChild(btn);
        });
        li.appendChild(pathSpan);
        li.appendChild(actions);
        filesEl.appendChild(li);
      });
      changesSection.style.display = files.length ? 'block' : 'none';
    }

    window.addEventListener('message', e => {
      const msg = e.data;
      if (msg.type === 'status') { statusEl.textContent = msg.text || ''; return; }
      if (msg.type === 'composerPrompt') {
        resultSection.style.display = 'block';
        responseEl.textContent = '(Running…)';
        renderFiles([]);
      }
      if (msg.type === 'composerResult') {
        resultSection.style.display = 'block';
        responseEl.textContent = msg.text || '(no output)';
        renderFiles(msg.files || []);
      }
      if (msg.type === 'composerUpdate') {
        renderFiles(msg.files || []);
      }
    });

    runBtn.addEventListener('click', () => {
      const q = promptEl.value.trim();
      if (!q) return;
      vscode.postMessage({ type: 'run', prompt: q });
    });

    applyAllBtn.addEventListener('click', () => vscode.postMessage({ type: 'applyAll' }));
    rejectAllBtn.addEventListener('click', () => vscode.postMessage({ type: 'rejectAll' }));
  </script>
</body>
</html>`;
}

export function deactivate() {}
