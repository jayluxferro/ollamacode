"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const vitest_1 = require("vitest");
const ollamacodeRunner_1 = require("./ollamacodeRunner");
vitest_1.vi.mock("vscode", () => ({
    workspace: {
        workspaceFolders: undefined,
        applyEdit: vitest_1.vi.fn(),
        openTextDocument: vitest_1.vi.fn(),
        getConfiguration: () => ({
            get: (_key, defaultValue) => defaultValue ?? "",
        }),
    },
    Uri: { joinPath: vitest_1.vi.fn() },
    Range: class {
    },
    WorkspaceEdit: class {
    },
    window: {},
    commands: {},
}));
(0, vitest_1.describe)("parseEditsFromOutput", () => {
    (0, vitest_1.it)("returns full stdout as text and empty edits when no markers", () => {
        const out = "Hello world\nNo edits here.";
        const { text, edits } = (0, ollamacodeRunner_1.parseEditsFromOutput)(out);
        (0, vitest_1.expect)(text).toBe(out.trim());
        (0, vitest_1.expect)(edits).toEqual([]);
    });
    (0, vitest_1.it)("parses valid edit block and strips it from text", () => {
        const before = "Here is the change:";
        const after = "Done.";
        const json = '{"edits":[{"path":"a.js","newText":"x"}]}';
        const out = `${before}\n<<OLLAMACODE_EDITS>>\n${json}\n<<END>>\n${after}`;
        const { text, edits } = (0, ollamacodeRunner_1.parseEditsFromOutput)(out);
        (0, vitest_1.expect)(text).toContain(before);
        (0, vitest_1.expect)(text).toContain(after);
        (0, vitest_1.expect)(text).not.toContain("<<OLLAMACODE_EDITS>>");
        (0, vitest_1.expect)(edits).toHaveLength(1);
        (0, vitest_1.expect)(edits[0].path).toBe("a.js");
        (0, vitest_1.expect)(edits[0].newText).toBe("x");
    });
    (0, vitest_1.it)("returns empty edits on invalid JSON between markers", () => {
        const out = "x\n<<OLLAMACODE_EDITS>>\nnot json\n<<END>>\ny";
        const { text, edits } = (0, ollamacodeRunner_1.parseEditsFromOutput)(out);
        (0, vitest_1.expect)(edits).toEqual([]);
        (0, vitest_1.expect)(text).toContain("x");
        (0, vitest_1.expect)(text).toContain("y");
    });
    (0, vitest_1.it)("returns empty edits when only start marker", () => {
        const out = "<<OLLAMACODE_EDITS>>\n{}";
        const { edits } = (0, ollamacodeRunner_1.parseEditsFromOutput)(out);
        (0, vitest_1.expect)(edits).toEqual([]);
    });
});
(0, vitest_1.describe)("getEditsByFile", () => {
    (0, vitest_1.it)("groups edits by path", () => {
        const edits = [
            { path: "a.js", newText: "1" },
            { path: "b.js", newText: "2" },
            { path: "a.js", newText: "3" },
        ];
        const byFile = (0, ollamacodeRunner_1.getEditsByFile)(edits);
        (0, vitest_1.expect)(byFile.size).toBe(2);
        (0, vitest_1.expect)(byFile.get("a.js")).toHaveLength(2);
        (0, vitest_1.expect)(byFile.get("b.js")).toHaveLength(1);
    });
    (0, vitest_1.it)("returns empty map for empty edits", () => {
        (0, vitest_1.expect)((0, ollamacodeRunner_1.getEditsByFile)([]).size).toBe(0);
    });
});
(0, vitest_1.describe)("applyEditsToContent", () => {
    (0, vitest_1.it)("returns original when no edits", () => {
        const content = "line1\nline2";
        (0, vitest_1.expect)((0, ollamacodeRunner_1.applyEditsToContent)(content, [])).toBe(content);
    });
    (0, vitest_1.it)("replaces range with newText", () => {
        const content = "abc\ndef\nghi";
        const edits = [
            {
                path: "x",
                range: { start: { line: 0, character: 0 }, end: { line: 0, character: 3 } },
                newText: "XYZ",
            },
        ];
        (0, vitest_1.expect)((0, ollamacodeRunner_1.applyEditsToContent)(content, edits)).toBe("XYZ\ndef\nghi");
    });
    (0, vitest_1.it)("applies multiple edits in reverse order (by start offset)", () => {
        const content = "0123456789";
        const edits = [
            { path: "x", range: { start: { line: 0, character: 2 }, end: { line: 0, character: 4 } }, newText: "B" },
            { path: "x", range: { start: { line: 0, character: 0 }, end: { line: 0, character: 2 } }, newText: "A" },
        ];
        (0, vitest_1.expect)((0, ollamacodeRunner_1.applyEditsToContent)(content, edits)).toBe("AB456789");
    });
});
//# sourceMappingURL=ollamacodeRunner.test.js.map