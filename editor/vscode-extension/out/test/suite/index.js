"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.run = run;
/**
 * Mocha test runner for extension E2E tests.
 * Runs all *.test.js in this directory (compiled from *.test.ts).
 */
const path = require("path");
const Mocha = require("mocha");
const glob_1 = require("glob");
function run() {
    const mocha = new Mocha({
        ui: "tdd",
        color: true,
        timeout: 30000,
    });
    const testsRoot = path.resolve(__dirname, ".");
    return new Promise((resolve, reject) => {
        (0, glob_1.glob)("**/*.test.js", { cwd: testsRoot })
            .then((files) => {
            files.forEach((f) => mocha.addFile(path.resolve(testsRoot, f)));
            try {
                mocha.run((failures) => {
                    if (failures > 0) {
                        reject(new Error(`${failures} tests failed.`));
                    }
                    else {
                        resolve();
                    }
                });
            }
            catch (err) {
                reject(err);
            }
        })
            .catch(reject);
    });
}
//# sourceMappingURL=index.js.map