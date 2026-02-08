"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.EDIT_PROTOCOL_SYSTEM_EXTRA = void 0;
exports.getConfig = getConfig;
exports.runCli = runCli;
exports.parseEditsFromOutput = parseEditsFromOutput;
exports.getEditsByFile = getEditsByFile;
exports.applyEditsToContent = applyEditsToContent;
exports.applyEdits = applyEdits;
const vscode = require("vscode");
const child_process_1 = require("child_process");
const EDITS_MARKER_START = "<<OLLAMACODE_EDITS>>";
const EDITS_MARKER_END = "<<END>>";
/** System extra instruction for the model to output edits in our protocol (injected when running from VS Code). */
exports.EDIT_PROTOCOL_SYSTEM_EXTRA = `When you suggest code changes to files in the user's workspace, output a single JSON block so the editor can apply them. Use this exact format, with no other text inside the block:
<<OLLAMACODE_EDITS>>
{"edits":[{"path":"relative/path/to/file","range":{"start":{"line":0,"character":0},"end":{"line":0,"character":0}},"newText":"content to insert or replace"}]}
<<END>>
- "path" is relative to the workspace root.
- "range" is 0-based (line, character). Omit "range" to replace the entire file.
- You can include multiple objects in "edits" for multiple files or regions.`;
function getConfig() {
    const config = vscode.workspace.getConfiguration("ollamacode");
    return {
        cliPath: config.get("cliPath", "ollamacode"),
        model: config.get("model", "qwen2.5-coder:32b"),
        mcpArgs: config.get("mcpArgs", ""),
    };
}
/** Run the ollamacode CLI with the given query; env includes OLLAMACODE_MCP_ARGS and optionally OLLAMACODE_SYSTEM_EXTRA. */
function runCli(query, options) {
    const { cliPath, model, mcpArgs } = getConfig();
    const parts = cliPath.trim().split(/\s+/);
    const cmd = parts[0];
    const baseArgs = parts.slice(1);
    const args = [...baseArgs, "-m", model];
    if (options.stream) {
        args.push("--stream");
    }
    args.push(query);
    const cwd = options.cwd ?? vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? process.cwd();
    const env = { ...process.env, ...options.env };
    if (mcpArgs.trim()) {
        env.OLLAMACODE_MCP_ARGS = mcpArgs.trim();
    }
    if (options.injectEditProtocol) {
        env.OLLAMACODE_SYSTEM_EXTRA = exports.EDIT_PROTOCOL_SYSTEM_EXTRA;
    }
    return new Promise((resolve) => {
        const proc = (0, child_process_1.spawn)(cmd, args, { cwd, shell: true, env });
        let stdout = "";
        let stderr = "";
        proc.stdout?.on("data", (chunk) => {
            const s = chunk.toString();
            stdout += s;
            options.onStreamChunk?.(s);
        });
        proc.stderr?.on("data", (chunk) => {
            stderr += chunk.toString();
        });
        proc.on("close", (code) => {
            const { text, edits } = parseEditsFromOutput(stdout);
            resolve({ stdout, stderr, code: code ?? 0, text, edits });
        });
    });
}
/** Parse <<OLLAMACODE_EDITS>>...<<END>> from stdout; return text (without block) and edits array. */
function parseEditsFromOutput(stdout) {
    const startIdx = stdout.indexOf(EDITS_MARKER_START);
    const endIdx = stdout.indexOf(EDITS_MARKER_END);
    if (startIdx === -1 || endIdx === -1 || endIdx <= startIdx) {
        return { text: stdout.trim(), edits: [] };
    }
    const jsonStr = stdout.slice(startIdx + EDITS_MARKER_START.length, endIdx).trim();
    const textBefore = stdout.slice(0, startIdx).trim();
    const textAfter = stdout.slice(endIdx + EDITS_MARKER_END.length).trim();
    const text = [textBefore, textAfter].filter(Boolean).join("\n\n");
    let edits = [];
    try {
        const parsed = JSON.parse(jsonStr);
        if (Array.isArray(parsed.edits)) {
            edits = parsed.edits;
        }
    }
    catch {
        // ignore parse errors
    }
    return { text, edits };
}
/** Group edits by file path. */
function getEditsByFile(edits) {
    const byFile = new Map();
    for (const e of edits) {
        const list = byFile.get(e.path) ?? [];
        list.push(e);
        byFile.set(e.path, list);
    }
    return byFile;
}
/** Apply a list of edits (for one file) to full text. Returns new content. */
function applyEditsToContent(fullText, fileEdits) {
    if (fileEdits.length === 0)
        return fullText;
    const lines = fullText.split("\n");
    const toOffset = (line, char) => {
        let o = 0;
        for (let i = 0; i < line && i < lines.length; i++)
            o += lines[i].length + 1;
        return o + Math.min(char, lines[line]?.length ?? 0);
    };
    const ranges = fileEdits.map((e) => ({
        start: e.range ? toOffset(e.range.start.line, e.range.start.character) : 0,
        end: e.range ? toOffset(e.range.end.line, e.range.end.character) : fullText.length,
        newText: e.newText,
    }));
    ranges.sort((a, b) => b.start - a.start);
    let result = fullText;
    for (const r of ranges) {
        result = result.slice(0, r.start) + r.newText + result.slice(r.end);
    }
    return result;
}
/** Apply edit specs to the workspace using vscode.workspace.applyEdit. */
async function applyEdits(edits) {
    const folder = vscode.workspace.workspaceFolders?.[0];
    if (!folder || edits.length === 0) {
        return false;
    }
    const workspaceEdit = new vscode.WorkspaceEdit();
    for (const e of edits) {
        const uri = vscode.Uri.joinPath(folder.uri, e.path);
        let range;
        if (e.range) {
            range = new vscode.Range(e.range.start.line, e.range.start.character, e.range.end.line, e.range.end.character);
        }
        else {
            const doc = await Promise.resolve(vscode.workspace.openTextDocument(uri)).catch(() => null);
            if (!doc) {
                range = new vscode.Range(0, 0, 0, 0);
            }
            else {
                const lastLine = doc.lineCount - 1;
                const lastChar = doc.lineAt(lastLine).text.length;
                range = new vscode.Range(0, 0, lastLine, lastChar);
            }
        }
        workspaceEdit.replace(uri, range, e.newText);
    }
    return vscode.workspace.applyEdit(workspaceEdit);
}
//# sourceMappingURL=ollamacodeRunner.js.map