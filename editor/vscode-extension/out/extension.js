"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const http = require("http");
const vscode = require("vscode");
const ollamacodeRunner_1 = require("./ollamacodeRunner");
const VIEW_TYPE = "ollamacode.chatView";
const COMPOSER_VIEW_TYPE = "ollamacode.composerView";
const CHAT_PARTICIPANT_ID = "ollamacode.chatParticipant";
const OLLAMA_TAGS_URL = "http://127.0.0.1:11434/api/tags";
let _logChannel;
function log(msg) {
    if (!_logChannel)
        _logChannel = vscode.window.createOutputChannel("OllamaCode");
    _logChannel.appendLine(`[${new Date().toISOString().slice(11, 23)}] ${msg}`);
}
function getNonce() {
    let text = "";
    const possible = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    for (let i = 0; i < 32; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
}
/** Fetch model names from Ollama API (GET /api/tags). Returns [] if Ollama is unreachable. Uses Node http so it works in all VS Code/Node versions. */
function fetchOllamaModels() {
    log(`Fetching models from ${OLLAMA_TAGS_URL}`);
    return new Promise((resolve) => {
        let settled = false;
        const finish = (names) => {
            if (settled)
                return;
            settled = true;
            log(names.length > 0 ? `Ollama models: ${names.length} (${names.slice(0, 3).join(", ")}${names.length > 3 ? "…" : ""})` : "Ollama unreachable or no models");
            resolve(names);
        };
        const req = http.get(OLLAMA_TAGS_URL, { timeout: 5000 }, (res) => {
            let body = "";
            res.on("data", (chunk) => { body += chunk.toString(); });
            res.on("end", () => {
                try {
                    const data = JSON.parse(body);
                    const names = data.models?.map((m) => m.name) ?? [];
                    finish(names.length > 0 ? names : []);
                }
                catch {
                    log("Ollama /api/tags response parse error");
                    finish([]);
                }
            });
        });
        req.on("error", (err) => {
            log(`Ollama request error: ${err.message}`);
            finish([]);
        });
        req.on("timeout", () => { req.destroy(); finish([]); });
    });
}
/** If injectCurrentFile is on and there is an active editor in the workspace, prepend current file context to the prompt. */
function promptWithFileContext(prompt) {
    const config = vscode.workspace.getConfiguration("ollamacode");
    if (!config.get("injectCurrentFile", true))
        return prompt;
    const editor = vscode.window.activeTextEditor;
    if (!editor)
        return prompt;
    const uri = editor.document.uri;
    const folder = vscode.workspace.getWorkspaceFolder(uri);
    if (!folder)
        return prompt;
    const relative = vscode.workspace.asRelativePath(uri);
    return `Current file: ${relative}\n\n${prompt}`;
}
/** Open side-by-side diff for each file in edits. */
async function openDiffsForEdits(edits) {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder)
        return;
    const byFile = (0, ollamacodeRunner_1.getEditsByFile)(edits);
    for (const [path, fileEdits] of byFile) {
        const uri = vscode.Uri.joinPath(folder.uri, path);
        const doc = await Promise.resolve(vscode.workspace.openTextDocument(uri)).catch(() => null);
        const current = doc ? doc.getText() : "";
        const newContent = (0, ollamacodeRunner_1.applyEditsToContent)(current, fileEdits);
        const lang = doc?.languageId ?? "plaintext";
        const newDoc = await vscode.workspace.openTextDocument({
            content: newContent,
            language: lang,
        });
        await vscode.commands.executeCommand("vscode.diff", uri, newDoc.uri, `OllamaCode: ${path}`);
    }
}
/** Open side-by-side diff for one file (path + its edits). */
async function openDiffForFile(path, fileEdits) {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder)
        return;
    const uri = vscode.Uri.joinPath(folder.uri, path);
    const doc = await Promise.resolve(vscode.workspace.openTextDocument(uri)).catch(() => null);
    const current = doc ? doc.getText() : "";
    const newContent = (0, ollamacodeRunner_1.applyEditsToContent)(current, fileEdits);
    const lang = doc?.languageId ?? "plaintext";
    const newDoc = await vscode.workspace.openTextDocument({
        content: newContent,
        language: lang,
    });
    await vscode.commands.executeCommand("vscode.diff", uri, newDoc.uri, `OllamaCode: ${path}`);
}
/** Apply only edits for the given path. */
function applyEditsForPath(edits, path) {
    return edits.filter((e) => e.path === path);
}
/** If review setting is on, show Apply all / Reject / Show diff; then apply or not. Else apply directly. */
async function applyEditsWithReview(edits) {
    const config = vscode.workspace.getConfiguration("ollamacode");
    const review = config.get("reviewEditsBeforeApply", false);
    if (!review || edits.length === 0) {
        return (0, ollamacodeRunner_1.applyEdits)(edits);
    }
    const choice = await vscode.window.showQuickPick([
        { label: "Apply all", value: "apply" },
        { label: "Reject", value: "reject" },
        { label: "Show diff first", value: "diff" },
    ], {
        title: `OllamaCode suggested ${edits.length} edit(s)`,
        placeHolder: "Apply all, reject, or show diff first",
    });
    if (!choice || choice.value === "reject") {
        return false;
    }
    if (choice.value === "diff") {
        await openDiffsForEdits(edits);
        const again = await vscode.window.showQuickPick([
            { label: "Apply all", value: "apply" },
            { label: "Reject", value: "reject" },
        ], { title: "Apply these edits?", placeHolder: "Apply all or Reject" });
        if (!again || again.value === "reject") {
            return false;
        }
    }
    return (0, ollamacodeRunner_1.applyEdits)(edits);
}
function activate(context) {
    context.subscriptions.push(vscode.commands.registerCommand("ollamacode.openPanel", () => {
        vscode.commands.executeCommand("workbench.view.extension.ollamacode");
    }));
    context.subscriptions.push(vscode.commands.registerCommand("ollamacode.selectModel", async () => {
        const models = await fetchOllamaModels();
        if (models.length === 0) {
            void vscode.window.showErrorMessage("OllamaCode: Could not fetch models. Is Ollama running? (http://127.0.0.1:11434)");
            return;
        }
        const config = vscode.workspace.getConfiguration("ollamacode");
        const current = config.get("model", "gpt-oss:20b");
        const picked = await vscode.window.showQuickPick(models, {
            title: "Select Ollama model",
            placeHolder: `Current: ${current}`,
            matchOnDescription: false,
        });
        if (picked) {
            await config.update("model", picked, vscode.ConfigurationTarget.Global);
            void vscode.window.showInformationMessage(`OllamaCode: Using model "${picked}"`);
        }
    }));
    const chatProvider = new OllamaCodeChatProvider(context);
    context.subscriptions.push(vscode.commands.registerCommand("ollamacode.chatWithSelection", async () => {
        const editor = vscode.window.activeTextEditor;
        const selection = editor?.document.getText(editor.selection)?.trim();
        await vscode.commands.executeCommand("workbench.view.extension.ollamacode");
        if (selection) {
            const query = `Selected code:\n\`\`\`\n${selection}\n\`\`\`\n\nWhat would you like to do with this code?`;
            chatProvider.runQuery(query);
        }
    }));
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(VIEW_TYPE, chatProvider));
    context.subscriptions.push(vscode.window.registerWebviewViewProvider(COMPOSER_VIEW_TYPE, new OllamaCodeComposerProvider(context)));
    // Chat Participant: @ollamacode in VS Code Chat (requires VS Code 1.109+)
    if (typeof vscode.chat.createChatParticipant === "function") {
        const participant = vscode.chat.createChatParticipant(CHAT_PARTICIPANT_ID, chatRequestHandler);
        context.subscriptions.push(participant);
    }
}
const chatRequestHandler = async (request, context, stream, token) => {
    const prompt = request.prompt;
    if (!prompt) {
        stream.markdown("No prompt provided.");
        return;
    }
    stream.progress("Running OllamaCode…");
    const config = vscode.workspace.getConfiguration("ollamacode");
    const useStream = config.get("streamResponse", true);
    const fullPrompt = promptWithFileContext(prompt);
    const result = await (0, ollamacodeRunner_1.runCli)(fullPrompt, {
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
class OllamaCodeChatProvider {
    _context;
    _view;
    _refreshChatView;
    _abortController;
    constructor(_context) {
        this._context = _context;
    }
    resolveWebviewView(webviewView, _context, _token) {
        log("Chat panel: resolveWebviewView");
        this._view = webviewView;
        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [],
        };
        const history = this._context.workspaceState.get("ollamacode.chatHistory") ?? [];
        let initialContentSent = false;
        const setHtml = (models, current, messages) => {
            webviewView.webview.html = getChatHtml(webviewView.webview, getNonce(), models, current, messages);
        };
        const setHtmlWithModels = async () => {
            let models = [];
            try {
                models = await fetchOllamaModels();
            }
            catch {
                models = [];
            }
            const config = vscode.workspace.getConfiguration("ollamacode");
            const current = config.get("model", "gpt-oss:20b");
            const messages = this._context.workspaceState.get("ollamacode.chatHistory") ?? [];
            log(`Chat panel: setting HTML with ${models.length} models, ${messages.length} messages`);
            setHtml(models, current, messages);
        };
        this._refreshChatView = setHtmlWithModels;
        const sendInitialContent = () => {
            if (initialContentSent)
                return;
            initialContentSent = true;
            log("Chat panel: sendInitialContent (loadHistory + pushModels)");
            this.postMessage({ type: "loadHistory", messages: history });
            void this.pushModelsToWebview();
        };
        void setHtmlWithModels();
        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                initialContentSent = false;
                void setHtmlWithModels();
                setTimeout(sendInitialContent, 600);
            }
        });
        webviewView.webview.onDidReceiveMessage(async (data) => {
            if (data.type === "webviewReady") {
                log("Chat panel: webviewReady received");
                sendInitialContent();
                // Always (re)send so webview gets models once its listener is ready (800ms may have fired before script ran).
                this.postMessage({ type: "loadHistory", messages: history });
                void this.pushModelsToWebview();
                return;
            }
            if (data.type === "getModels") {
                log("Chat panel: getModels received");
                let models = [];
                try {
                    models = await fetchOllamaModels();
                }
                catch {
                    models = [];
                }
                const config = vscode.workspace.getConfiguration("ollamacode");
                const current = config.get("model", "gpt-oss:20b");
                this.postMessage({ type: "models", models, current });
            }
            if (data.type === "send" && data.query) {
                log(`Chat: received send query (${data.query.length} chars)`);
                this.postMessage({ type: "status", text: "Sending…" });
                try {
                    await this.runQuery(data.query);
                    log("Chat: runQuery completed");
                }
                catch (e) {
                    const err = e;
                    log(`Chat: runQuery error: ${err.message}`);
                    this.postMessage({ type: "append", role: "assistant", text: `Error: ${err.message}` });
                    void vscode.window.showErrorMessage(`OllamaCode: ${err.message}`);
                }
            }
            if (data.type === "setModel" && data.model) {
                const config = vscode.workspace.getConfiguration("ollamacode");
                await config.update("model", data.model, vscode.ConfigurationTarget.Global);
                this.postMessage({ type: "modelSet", model: data.model });
            }
            if (data.type === "stop") {
                log("Chat: stop requested");
                this._abortController?.abort();
            }
            if (data.type === "newChat") {
                log("Chat: new chat requested");
                await this._context.workspaceState.update("ollamacode.chatHistory", []);
                this.postMessage({ type: "loadHistory", messages: [] });
                void this._refreshChatView?.();
            }
        });
        const sendLoadHistoryAndModels = () => {
            this.postMessage({ type: "loadHistory", messages: history });
            void this.pushModelsToWebview();
        };
        setTimeout(sendInitialContent, 800);
        setTimeout(sendLoadHistoryAndModels, 1500);
        setTimeout(sendLoadHistoryAndModels, 3000);
    }
    async runQuery(query) {
        log("Chat: runQuery start");
        this._abortController = new AbortController();
        this.postMessage({ type: "status", text: "Running OllamaCode…" });
        const config = vscode.workspace.getConfiguration("ollamacode");
        const useStream = config.get("streamResponse", true);
        const fullPrompt = promptWithFileContext(query);
        this.postMessage({ type: "streamStart" });
        const result = await (0, ollamacodeRunner_1.runCli)(fullPrompt, {
            injectEditProtocol: true,
            stream: useStream,
            onStreamChunk: useStream
                ? (chunk) => this.postMessage({ type: "streamChunk", text: chunk })
                : undefined,
            signal: this._abortController.signal,
        });
        this._abortController = undefined;
        this.postMessage({ type: "streamEnd" });
        this.postMessage({ type: "status", text: "" });
        let assistantText = "";
        if (result.code === 143) {
            assistantText = (result.text || "").trim() || "(stopped)";
            if (!result.text?.trim())
                this.postMessage({ type: "append", role: "assistant", text: "(stopped)" });
        }
        else if (result.code !== 0 && result.stderr) {
            assistantText = `Error (exit ${result.code}):\n${result.stderr}`;
            this.postMessage({ type: "append", role: "assistant", text: assistantText });
        }
        else {
            if (!useStream) {
                assistantText = result.text || "(no output)";
                this.postMessage({ type: "append", role: "assistant", text: assistantText });
            }
            else if (result.edits.length > 0 || result.text) {
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
        const prev = this._context.workspaceState.get("ollamacode.chatHistory") ?? [];
        const next = [...prev, { role: "user", content: query }, { role: "assistant", content: assistantText }].slice(-maxHistory * 2);
        this._context.workspaceState.update("ollamacode.chatHistory", next);
        if (assistantText) {
            log("--- Chat response ---");
            log(assistantText.slice(0, 2000) + (assistantText.length > 2000 ? "\n…" : ""));
        }
        void this._refreshChatView?.();
    }
    postMessage(msg) {
        this._view?.webview.postMessage(msg);
    }
    async pushModelsToWebview() {
        let models = [];
        try {
            models = await fetchOllamaModels();
        }
        catch (e) {
            log(`pushModelsToWebview fetch error: ${e.message}`);
            models = [];
        }
        const config = vscode.workspace.getConfiguration("ollamacode");
        const current = config.get("model", "gpt-oss:20b");
        log(`Pushing ${models.length} models to webview, current: ${current}`);
        this.postMessage({ type: "models", models, current });
    }
}
/** Composer: multi-file task → proposed changes list → Preview / Apply / Reject per file or all. */
class OllamaCodeComposerProvider {
    _context;
    _view;
    _pendingEdits = [];
    _pendingText = "";
    constructor(_context) {
        this._context = _context;
    }
    resolveWebviewView(webviewView, _context, _token) {
        this._view = webviewView;
        webviewView.webview.options = { enableScripts: true, localResourceRoots: [] };
        const nonce = getNonce();
        webviewView.webview.html = getComposerHtml(webviewView.webview, nonce);
        webviewView.webview.onDidReceiveMessage(async (data) => {
            if (data.type === "run" && data.prompt) {
                await this.runComposer(data.prompt);
            }
            else if (data.type === "previewFile" && data.path) {
                await this.previewFile(data.path);
            }
            else if (data.type === "applyFile" && data.path) {
                await this.applyFile(data.path);
            }
            else if (data.type === "rejectFile" && data.path) {
                this.rejectFile(data.path);
            }
            else if (data.type === "applyAll") {
                await this.applyAll();
            }
            else if (data.type === "rejectAll") {
                this.rejectAll();
            }
        });
    }
    async runComposer(prompt) {
        this.postMessage({ type: "status", text: "Running OllamaCode…" });
        this.postMessage({ type: "composerPrompt", text: prompt });
        const fullPrompt = promptWithFileContext(prompt);
        const result = await (0, ollamacodeRunner_1.runCli)(fullPrompt, { injectEditProtocol: true });
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
        const byFile = (0, ollamacodeRunner_1.getEditsByFile)(result.edits);
        const files = Array.from(byFile.keys());
        this.postMessage({
            type: "composerResult",
            text: this._pendingText,
            files,
        });
    }
    async previewFile(path) {
        const byFile = (0, ollamacodeRunner_1.getEditsByFile)(this._pendingEdits);
        const fileEdits = byFile.get(path);
        if (fileEdits) {
            await openDiffForFile(path, fileEdits);
        }
    }
    async applyFile(path) {
        const toApply = applyEditsForPath(this._pendingEdits, path);
        if (toApply.length === 0)
            return;
        const ok = await (0, ollamacodeRunner_1.applyEdits)(toApply);
        if (ok) {
            this._pendingEdits = this._pendingEdits.filter((e) => e.path !== path);
            this.postMessage({ type: "composerUpdate", files: Array.from((0, ollamacodeRunner_1.getEditsByFile)(this._pendingEdits).keys()) });
        }
    }
    rejectFile(path) {
        this._pendingEdits = this._pendingEdits.filter((e) => e.path !== path);
        this.postMessage({ type: "composerUpdate", files: Array.from((0, ollamacodeRunner_1.getEditsByFile)(this._pendingEdits).keys()) });
    }
    async applyAll() {
        if (this._pendingEdits.length === 0)
            return;
        const ok = await (0, ollamacodeRunner_1.applyEdits)(this._pendingEdits);
        if (ok) {
            this._pendingEdits = [];
            this.postMessage({ type: "composerUpdate", files: [] });
        }
    }
    rejectAll() {
        this._pendingEdits = [];
        this.postMessage({ type: "composerUpdate", files: [] });
    }
    postMessage(msg) {
        this._view?.webview.postMessage(msg);
    }
}
function escapeHtml(s) {
    return s
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}
/** Minimal markdown: code blocks (with copy), inline code, bold, italic, links. Safe for HTML when input is plain text. */
function renderMarkdownToHtml(text) {
    const codeBlocks = [];
    let s = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_m, _lang, code) => {
        codeBlocks.push(code);
        return `\x00CB${codeBlocks.length - 1}\x00`;
    });
    s = escapeHtml(s)
        .replace(/\n/g, "<br>")
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/\*(.+?)\*/g, "<em>$1</em>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/\x00CB(\d+)\x00/g, (_, i) => {
        const code = escapeHtml(codeBlocks[Number(i)] ?? "");
        const attr = code.replace(/"/g, "&quot;").replace(/\n/g, "&#10;");
        return `<div class="code-block"><button class="copy-btn" data-code="${attr}" title="Copy">Copy</button><pre><code>${code}</code></pre></div>`;
    });
    return `<div class="msg-content">${s}</div>`;
}
function getChatHtml(webview, nonce, initialModels, initialCurrent, initialMessages) {
    const csp = `default-src 'none'; style-src 'nonce-${nonce}' 'unsafe-inline' ${webview.cspSource}; script-src 'nonce-${nonce}' ${webview.cspSource};`;
    const hasInitialModels = Array.isArray(initialModels) && initialModels.length > 0;
    const current = initialCurrent || "";
    let selectOptionsHtml;
    if (hasInitialModels) {
        const opts = initialModels.map((name) => `<option value="${escapeHtml(name)}"${name === current ? " selected" : ""}>${escapeHtml(name)}</option>`);
        if (current && !initialModels.includes(current)) {
            opts.unshift(`<option value="${escapeHtml(current)}" selected>${escapeHtml(current)} (current)</option>`);
        }
        selectOptionsHtml = opts.join("");
    }
    else {
        selectOptionsHtml = '<option value="">Loading…</option>';
    }
    const messagesHtml = Array.isArray(initialMessages) && initialMessages.length > 0
        ? initialMessages
            .map((m) => {
            const role = m.role || "assistant";
            const raw = m.content || "";
            const body = role === "assistant"
                ? renderMarkdownToHtml(raw)
                : escapeHtml(raw).replace(/\n/g, "<br>");
            return `<div class="msg ${escapeHtml(role)}">${body}</div>`;
        })
            .join("")
        : "";
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp.replace(/"/g, "&quot;")}">
  <style nonce="${nonce}">
    * { box-sizing: border-box; }
    body { margin: 0; padding: 8px; min-height: 200px; font-family: var(--vscode-font-family); font-size: var(--vscode-font-size); color: var(--vscode-foreground); background: var(--vscode-editor-background, #1e1e1e); }
    #messages { min-height: 120px; max-height: 60vh; overflow-y: auto; margin-bottom: 8px; }
    .msg { margin: 6px 0; padding: 8px 10px; border-radius: 6px; word-break: break-word; }
    .msg.user { background: var(--vscode-input-background); white-space: pre-wrap; }
    .msg.assistant { background: var(--vscode-editor-inactiveSelectionBackground); }
    .msg.assistant .msg-content { line-height: 1.5; }
    .msg.assistant .msg-content p { margin: 0.5em 0; }
    .msg.assistant .msg-content pre { margin: 8px 0; overflow-x: auto; }
    .msg.assistant .msg-content code { font-family: var(--vscode-editor-font-family); background: var(--vscode-textCodeBlock-background); padding: 2px 4px; border-radius: 4px; font-size: 0.9em; }
    .msg.assistant .msg-content pre code { padding: 8px; display: block; }
    .code-block { position: relative; margin: 8px 0; }
    .code-block .copy-btn { position: absolute; top: 4px; right: 4px; padding: 4px 8px; font-size: 0.75em; }
    #status { font-size: 0.9em; color: var(--vscode-descriptionForeground); margin-bottom: 6px; min-height: 1.2em; }
    #toolbar { display: flex; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }
    #model-row { display: flex; align-items: center; gap: 8px; }
    #model-row label { font-size: 0.9em; color: var(--vscode-descriptionForeground); }
    #model-select { flex: 1; min-width: 120px; padding: 4px 8px; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); border-radius: 4px; }
    #input-row { display: flex; flex-direction: column; gap: 6px; }
    #input-wrap { display: flex; gap: 6px; align-items: flex-end; }
    textarea#input { flex: 1; min-height: 44px; max-height: 120px; padding: 8px 10px; resize: none; background: var(--vscode-input-background); color: var(--vscode-input-foreground); border: 1px solid var(--vscode-input-border); border-radius: 4px; font-family: inherit; font-size: inherit; }
    button { padding: 6px 12px; background: var(--vscode-button-background); color: var(--vscode-button-foreground); border: none; border-radius: 4px; cursor: pointer; white-space: nowrap; }
    button:hover { background: var(--vscode-button-hoverBackground); }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    button.secondary { background: var(--vscode-button-secondaryBackground); color: var(--vscode-button-secondaryForeground); }
    button.secondary:hover { background: var(--vscode-button-secondaryHoverBackground); }
  </style>
</head>
<body>
  <div id="status"></div>
  <div id="toolbar">
    <div id="model-row">
      <label for="model-select">Model:</label>
      <select id="model-select" title="Ollama model. Select to set as default.">
        ${selectOptionsHtml}
      </select>
    </div>
    <button type="button" id="new-chat-btn" class="secondary" title="Start new chat">New Chat</button>
    <button type="button" id="stop-btn" class="secondary" title="Stop generation" disabled>Stop</button>
  </div>
  <div id="messages">${messagesHtml}</div>
  <form id="form">
    <div id="input-row">
      <div id="input-wrap">
        <textarea id="input" placeholder="Ask OllamaCode… (Enter to send, Shift+Enter for new line)" rows="1"></textarea>
        <button type="submit" id="send-btn">Send</button>
      </div>
    </div>
  </form>
  <script nonce="${nonce}">
    const vscode = acquireVsCodeApi();
    const statusEl = document.getElementById('status');
    const messagesEl = document.getElementById('messages');
    const formEl = document.getElementById('form');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('send-btn');
    const modelSelect = document.getElementById('model-select');
    const newChatBtn = document.getElementById('new-chat-btn');
    const stopBtn = document.getElementById('stop-btn');
    let modelsReceived = ${hasInitialModels ? "true" : "false"};

    function escapeHtml(s) {
      return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }
    function renderMarkdown(text) {
      const codeBlocks = [];
      const re = /\`\`\`(\\w*)\\n?([\\s\\S]*?)\`\`\`/g;
      let s = text.replace(re, function(_, lang, code) {
        codeBlocks.push(code);
        return '\u200B@' + (codeBlocks.length - 1) + '@\u200B';
      });
      s = escapeHtml(s).replace(/\\n/g, '<br>')
        .replace(/\\*\\*([^*]+)\\*\\*/g, '<strong>$1</strong>')
        .replace(/\\*([^*]+)\\*/g, '<em>$1</em>')
        .replace(/\`([^\`]+)\`/g, '<code>$1</code>')
        .replace(/\\[([^\\]]+)\\]\\(([^)]+)\\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
      s = s.replace(/\u200B@(\\d+)@\u200B/g, function(_, i) {
        const code = escapeHtml(codeBlocks[Number(i)] || '');
        const attr = code.replace(/"/g, '&quot;').replace(/\n/g, '&#10;');
        return '<div class="code-block"><button class="copy-btn" data-code="' + attr + '" title="Copy">Copy</button><pre><code>' + code + '</code></pre></div>';
      });
      return '<div class="msg-content">' + s + '</div>';
    }
    function attachCopyButtons(container) {
      if (!container) return;
      container.querySelectorAll('.copy-btn').forEach(function(btn) {
        if (btn.dataset.copied) return;
        btn.dataset.copied = '1';
        btn.addEventListener('click', function() {
          let code = btn.getAttribute('data-code') || '';
          code = code.replace(/&#10;/g, '\n').replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
          navigator.clipboard.writeText(code).then(function() {
            const t = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(function() { btn.textContent = t; }, 1500);
          });
        });
      });
    }

    setTimeout(() => {
      if (!modelsReceived && modelSelect.options.length <= 1 && modelSelect.options[0]?.value === '') {
        modelSelect.innerHTML = '';
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No models — see Output → OllamaCode';
        modelSelect.appendChild(opt);
      }
    }, 5000);

    modelSelect.addEventListener('change', () => {
      const v = modelSelect.value;
      if (v) vscode.postMessage({ type: 'setModel', model: v });
    });

    let streamTarget = null;
    window.addEventListener('message', e => {
      const msg = e.data;
      if (msg.type === 'status') {
        statusEl.textContent = msg.text || '';
        if (!msg.text) {
          inputEl.disabled = false;
          sendBtn.disabled = false;
          stopBtn.disabled = true;
        }
        return;
      }
      if (msg.type === 'streamStart') {
        const div = document.createElement('div');
        div.className = 'msg assistant';
        messagesEl.appendChild(div);
        streamTarget = div;
        stopBtn.disabled = false;
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return;
      }
      if (msg.type === 'streamChunk' && streamTarget) {
        streamTarget.textContent += msg.text || '';
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return;
      }
      if (msg.type === 'streamEnd') {
        if (streamTarget) {
          const raw = streamTarget.textContent || '';
          streamTarget.innerHTML = renderMarkdown(raw);
          attachCopyButtons(streamTarget);
        }
        streamTarget = null;
        inputEl.disabled = false;
        sendBtn.disabled = false;
        stopBtn.disabled = true;
        return;
      }
      if (msg.type === 'streamResult') {
        if (msg.editsCount > 0) {
          const last = messagesEl.querySelector('.msg.assistant:last-child');
          if (last) {
            const p = document.createElement('p');
            p.textContent = '[Applied ' + msg.editsCount + ' edit(s) to the workspace.]';
            p.style.marginTop = '8px';
            last.appendChild(p);
          }
        }
        streamTarget = null;
        inputEl.disabled = false;
        sendBtn.disabled = false;
        stopBtn.disabled = true;
        return;
      }
      if (msg.type === 'append') {
        const div = document.createElement('div');
        div.className = 'msg ' + msg.role;
        if (msg.role === 'assistant') {
          div.innerHTML = renderMarkdown(msg.text || '');
          attachCopyButtons(div);
        } else {
          div.style.whiteSpace = 'pre-wrap';
          div.textContent = msg.text || '';
        }
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        if (msg.role === 'assistant') {
          inputEl.disabled = false;
          sendBtn.disabled = false;
          stopBtn.disabled = true;
        }
      }
      if (msg.type === 'loadHistory' && Array.isArray(msg.messages)) {
        messagesEl.innerHTML = '';
        msg.messages.forEach(function(m) {
          const div = document.createElement('div');
          div.className = 'msg ' + (m.role || 'assistant');
          if (m.role === 'assistant') {
            div.innerHTML = renderMarkdown(m.content || '');
            attachCopyButtons(div);
          } else {
            div.style.whiteSpace = 'pre-wrap';
            div.textContent = m.content || '';
          }
          messagesEl.appendChild(div);
        });
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }
      if (msg.type === 'models') {
        modelsReceived = true;
        const list = msg.models || [];
        const current = msg.current || '';
        modelSelect.innerHTML = '';
        if (list.length === 0) {
          const opt = document.createElement('option');
          opt.value = '';
          opt.textContent = 'Ollama not available';
          modelSelect.appendChild(opt);
        } else {
          list.forEach(name => {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            if (name === current) opt.selected = true;
            modelSelect.appendChild(opt);
          });
          if (current && !list.includes(current)) {
            const opt = document.createElement('option');
            opt.value = current;
            opt.textContent = current + ' (current)';
            opt.selected = true;
            modelSelect.insertBefore(opt, modelSelect.firstChild);
          }
        }
      }
      if (msg.type === 'modelSet') {
        statusEl.textContent = 'Model set to ' + (msg.model || '');
        setTimeout(() => { statusEl.textContent = ''; }, 2000);
      }
    });

    inputEl.addEventListener('keydown', function(e) {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        formEl.requestSubmit();
      }
    });

    formEl.addEventListener('submit', function(e) {
      e.preventDefault();
      const q = inputEl.value.trim();
      if (!q) return;
      const userDiv = document.createElement('div');
      userDiv.className = 'msg user';
      userDiv.style.whiteSpace = 'pre-wrap';
      userDiv.textContent = q;
      messagesEl.appendChild(userDiv);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      statusEl.textContent = 'Sending…';
      inputEl.value = '';
      inputEl.disabled = true;
      sendBtn.disabled = true;
      stopBtn.disabled = false;
      vscode.postMessage({ type: 'send', query: q });
    });

    newChatBtn.addEventListener('click', function() {
      vscode.postMessage({ type: 'newChat' });
    });

    stopBtn.addEventListener('click', function() {
      vscode.postMessage({ type: 'stop' });
    });

    attachCopyButtons(messagesEl);

    vscode.postMessage({ type: 'webviewReady' });
    setTimeout(() => vscode.postMessage({ type: 'getModels' }), 1000);
    setTimeout(() => vscode.postMessage({ type: 'getModels' }), 2500);
  </script>
</body>
</html>`;
}
function getComposerHtml(webview, nonce) {
    const csp = `default-src 'none'; style-src 'nonce-${nonce}' 'unsafe-inline' ${webview.cspSource}; script-src 'nonce-${nonce}' ${webview.cspSource};`;
    return `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="${csp.replace(/"/g, "&quot;")}">
  <style nonce="${nonce}">
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
  <script nonce="${nonce}">
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
function deactivate() { }
//# sourceMappingURL=extension.js.map