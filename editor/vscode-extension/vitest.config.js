"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const config_1 = require("vitest/config");
const path = require("path");
exports.default = (0, config_1.defineConfig)({
    test: {
        environment: "node",
        include: ["src/**/*.test.ts"],
        globals: true,
    },
    resolve: {
        alias: {
            "@": path.resolve(__dirname, "./src"),
        },
    },
});
//# sourceMappingURL=vitest.config.js.map