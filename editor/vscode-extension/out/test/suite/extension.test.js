"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * E2E test: extension loads and Open Chat Panel command can be executed.
 */
const assert = require("assert");
const vscode = require("vscode");
suite("OllamaCode Extension E2E", () => {
    test("OllamaCode: Open Chat Panel command executes without throwing", async () => {
        await assert.doesNotReject(async () => {
            await vscode.commands.executeCommand("ollamacode.openPanel");
        });
    });
});
//# sourceMappingURL=extension.test.js.map