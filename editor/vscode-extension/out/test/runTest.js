"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
/**
 * E2E test runner: launches VS Code Extension Development Host and runs test/suite.
 * Run with: npm run test:e2e
 * Requires: no other instance of VS Code (stable) running when run from CLI.
 */
const path = require("path");
const test_electron_1 = require("@vscode/test-electron");
async function main() {
    try {
        const extensionDevelopmentPath = path.resolve(__dirname, "../../");
        const extensionTestsPath = path.resolve(__dirname, "./suite/index.js");
        await (0, test_electron_1.runTests)({
            extensionDevelopmentPath,
            extensionTestsPath,
            launchArgs: ["--disable-extensions"],
        });
    }
    catch (err) {
        console.error(err);
        console.error("Failed to run extension E2E tests");
        process.exit(1);
    }
}
main();
//# sourceMappingURL=runTest.js.map