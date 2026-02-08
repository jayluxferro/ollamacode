/**
 * E2E test runner: launches VS Code Extension Development Host and runs test/suite.
 * Run with: npm run test:e2e
 * Requires: no other instance of VS Code (stable) running when run from CLI.
 */
import * as path from "path";
import { runTests } from "@vscode/test-electron";

async function main(): Promise<void> {
  try {
    const extensionDevelopmentPath = path.resolve(__dirname, "../../");
    const extensionTestsPath = path.resolve(__dirname, "./suite/index.js");

    await runTests({
      extensionDevelopmentPath,
      extensionTestsPath,
      launchArgs: ["--disable-extensions"],
    });
  } catch (err) {
    console.error(err);
    console.error("Failed to run extension E2E tests");
    process.exit(1);
  }
}

main();
