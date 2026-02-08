/**
 * E2E test: extension loads and Open Chat Panel command can be executed.
 */
import * as assert from "assert";
import * as vscode from "vscode";

suite("OllamaCode Extension E2E", () => {
  test("OllamaCode: Open Chat Panel command executes without throwing", async () => {
    await assert.doesNotReject(async () => {
      await vscode.commands.executeCommand("ollamacode.openPanel");
    });
  });
});
