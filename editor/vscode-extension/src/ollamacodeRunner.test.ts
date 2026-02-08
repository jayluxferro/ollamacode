import { describe, it, expect, vi } from "vitest";
import {
  parseEditsFromOutput,
  getEditsByFile,
  applyEditsToContent,
  type EditSpec,
} from "./ollamacodeRunner";

vi.mock("vscode", () => ({
  workspace: {
    workspaceFolders: undefined,
    applyEdit: vi.fn(),
    openTextDocument: vi.fn(),
    getConfiguration: () => ({
      get: (_key: string, defaultValue?: unknown) => defaultValue ?? "",
    }),
  },
  Uri: { joinPath: vi.fn() },
  Range: class {},
  WorkspaceEdit: class {},
  window: {},
  commands: {},
}));

describe("parseEditsFromOutput", () => {
  it("returns full stdout as text and empty edits when no markers", () => {
    const out = "Hello world\nNo edits here.";
    const { text, edits } = parseEditsFromOutput(out);
    expect(text).toBe(out.trim());
    expect(edits).toEqual([]);
  });

  it("parses valid edit block and strips it from text", () => {
    const before = "Here is the change:";
    const after = "Done.";
    const json = '{"edits":[{"path":"a.js","newText":"x"}]}';
    const out = `${before}\n<<OLLAMACODE_EDITS>>\n${json}\n<<END>>\n${after}`;
    const { text, edits } = parseEditsFromOutput(out);
    expect(text).toContain(before);
    expect(text).toContain(after);
    expect(text).not.toContain("<<OLLAMACODE_EDITS>>");
    expect(edits).toHaveLength(1);
    expect(edits[0].path).toBe("a.js");
    expect(edits[0].newText).toBe("x");
  });

  it("returns empty edits on invalid JSON between markers", () => {
    const out = "x\n<<OLLAMACODE_EDITS>>\nnot json\n<<END>>\ny";
    const { text, edits } = parseEditsFromOutput(out);
    expect(edits).toEqual([]);
    expect(text).toContain("x");
    expect(text).toContain("y");
  });

  it("returns empty edits when only start marker", () => {
    const out = "<<OLLAMACODE_EDITS>>\n{}";
    const { edits } = parseEditsFromOutput(out);
    expect(edits).toEqual([]);
  });
});

describe("getEditsByFile", () => {
  it("groups edits by path", () => {
    const edits: EditSpec[] = [
      { path: "a.js", newText: "1" },
      { path: "b.js", newText: "2" },
      { path: "a.js", newText: "3" },
    ];
    const byFile = getEditsByFile(edits);
    expect(byFile.size).toBe(2);
    expect(byFile.get("a.js")).toHaveLength(2);
    expect(byFile.get("b.js")).toHaveLength(1);
  });

  it("returns empty map for empty edits", () => {
    expect(getEditsByFile([]).size).toBe(0);
  });
});

describe("applyEditsToContent", () => {
  it("returns original when no edits", () => {
    const content = "line1\nline2";
    expect(applyEditsToContent(content, [])).toBe(content);
  });

  it("replaces range with newText", () => {
    const content = "abc\ndef\nghi";
    const edits: EditSpec[] = [
      {
        path: "x",
        range: { start: { line: 0, character: 0 }, end: { line: 0, character: 3 } },
        newText: "XYZ",
      },
    ];
    expect(applyEditsToContent(content, edits)).toBe("XYZ\ndef\nghi");
  });

  it("applies multiple edits in reverse order (by start offset)", () => {
    const content = "0123456789";
    const edits: EditSpec[] = [
      { path: "x", range: { start: { line: 0, character: 2 }, end: { line: 0, character: 4 } }, newText: "B" },
      { path: "x", range: { start: { line: 0, character: 0 }, end: { line: 0, character: 2 } }, newText: "A" },
    ];
    expect(applyEditsToContent(content, edits)).toBe("AB456789");
  });
});
