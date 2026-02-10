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
}

function getConfig() {
  const config = vscode.workspace.getConfiguration("ollamacode");
  const baseUrl = (config.get<string>("baseUrl") || "http://localhost:8000").replace(/\/$/, "");
  const apiKey = config.get<string>("apiKey") || "";
  return { baseUrl, apiKey };
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

  async function chat(includeSelection: boolean): Promise<void> {
    const { baseUrl, apiKey } = getConfig();
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

    const body: Record<string, unknown> = { message };
    if (file) body.file = file;
    if (lines) body.lines = lines;

    output.appendLine(`POST ${baseUrl}/chat`);
    try {
      const raw = await httpPost(
        `${baseUrl}/chat`,
        JSON.stringify(body),
        getHeaders(apiKey)
      );
      const data = JSON.parse(raw) as ChatResponse;
      if (data.error) {
        output.appendLine(`Error: ${data.error}`);
        vscode.window.showErrorMessage(`OllamaCode: ${data.error}`);
        return;
      }
      if (data.content) {
        output.appendLine(data.content);
        output.show();
      }
      if (data.edits && data.edits.length > 0) {
        const apply = "Apply edits";
        const choice = await vscode.window.showInformationMessage(
          `OllamaCode returned ${data.edits.length} edit(s).`,
          apply,
          "Dismiss"
        );
        if (choice === apply) {
          await applyEditsToWorkspace(data.edits);
        }
      }
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      output.appendLine(`Request failed: ${err}`);
      vscode.window.showErrorMessage(
        `OllamaCode: request failed. Is \`ollamacode serve\` running at ${baseUrl}?`
      );
    }
  }

  async function chatStream(includeSelection: boolean): Promise<void> {
    const { baseUrl, apiKey } = getConfig();
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

    const body: Record<string, unknown> = { message };
    if (file) body.file = file;
    if (lines) body.lines = lines;

    output.clear();
    output.appendLine(`POST ${baseUrl}/chat/stream`);
    output.show();
    try {
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
                "Apply edits",
                "Dismiss"
              ).then((choice) => {
                if (choice === "Apply edits") {
                  applyEditsToWorkspace(event.edits!);
                }
              });
            }
          }
          if (event.type === "error" && event.error) {
            output.appendLine(`\nError: ${event.error}`);
            vscode.window.showErrorMessage(`OllamaCode: ${event.error}`);
          }
        }
      );
    } catch (e) {
      const err = e instanceof Error ? e.message : String(e);
      output.appendLine(`\nRequest failed: ${err}`);
      vscode.window.showErrorMessage(
        `OllamaCode: request failed. Is \`ollamacode serve\` running at ${baseUrl}?`
      );
    }
  }

  context.subscriptions.push(
    vscode.commands.registerCommand("ollamacode.chat", () => chat(false)),
    vscode.commands.registerCommand("ollamacode.chatWithSelection", () => chat(true)),
    vscode.commands.registerCommand("ollamacode.chatStream", () => chatStream(false)),
    vscode.commands.registerCommand("ollamacode.chatStreamWithSelection", () => chatStream(true)),
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
    })
  );
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
