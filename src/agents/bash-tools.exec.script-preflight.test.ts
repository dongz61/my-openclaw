import fs from "node:fs/promises";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { withTempDir } from "../test-utils/temp-dir.js";
import { createExecTool } from "./bash-tools.exec.js";

const isWin = process.platform === "win32";

const describeNonWin = isWin ? describe.skip : describe;

describeNonWin("exec script preflight", () => {
  it("returns a failed tool result for SWE-bench environment mutation commands", async () => {
    await withTempDir("openclaw-exec-env-guard-", async (tmp) => {
      const previous = process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
      process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = "1";
      try {
        const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });
        const result = await tool.execute("call-env-guard", {
          command: "python -m pip install -e .",
          workdir: tmp,
        });
        const text = result.content.find((block) => block.type === "text")?.text ?? "";

        expect(result.details.status).toBe("failed");
        expect(text).toContain("exec blocked by SWE-bench environment guard");
        expect(text).toContain("Do not install dependencies");
      } finally {
        if (previous === undefined) {
          delete process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
        } else {
          process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = previous;
        }
      }
    });
  });

  it("blocks setup.py build commands behind shell wrappers under the SWE-bench guard", async () => {
    await withTempDir("openclaw-exec-env-guard-", async (tmp) => {
      const previous = process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
      process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = "1";
      try {
        const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });
        const result = await tool.execute("call-env-guard-setup", {
          command: "cd astropy && python setup.py build_ext --inplace 2>&1 | tail -50",
          workdir: tmp,
        });
        const text = result.content.find((block) => block.type === "text")?.text ?? "";

        expect(result.details.status).toBe("failed");
        expect(text).toContain("exec blocked by SWE-bench environment guard");
        expect(text).toContain("Python setup/build command");
        expect(text).toContain("packaging/build/setup commands");
      } finally {
        if (previous === undefined) {
          delete process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
        } else {
          process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = previous;
        }
      }
    });
  });

  it("blocks setup.py build commands inside bash -lc wrappers under the SWE-bench guard", async () => {
    await withTempDir("openclaw-exec-env-guard-", async (tmp) => {
      const previous = process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
      process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = "1";
      try {
        const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });
        const result = await tool.execute("call-env-guard-setup-bash", {
          command:
            'bash -lc "cd /home/ubuntu/桌面/my-openclaw/experiment/swebench/workspaces/astropy__astropy-7746 && python setup.py build_ext --inplace 2>&1 | tail -50"',
          workdir: tmp,
        });
        const text = result.content.find((block) => block.type === "text")?.text ?? "";

        expect(result.details.status).toBe("failed");
        expect(text).toContain("exec blocked by SWE-bench environment guard");
        expect(text).toContain("Python setup/build command");
      } finally {
        if (previous === undefined) {
          delete process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
        } else {
          process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = previous;
        }
      }
    });
  });

  it("blocks env-prefixed setup.py build commands under the SWE-bench guard", async () => {
    await withTempDir("openclaw-exec-env-guard-", async (tmp) => {
      const previous = process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
      process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = "1";
      try {
        const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });
        const result = await tool.execute("call-env-guard-setup-env", {
          command: "cd astropy && PYTHONPATH=. python ./setup.py build_ext --inplace",
          workdir: tmp,
        });
        const text = result.content.find((block) => block.type === "text")?.text ?? "";

        expect(result.details.status).toBe("failed");
        expect(text).toContain("exec blocked by SWE-bench environment guard");
        expect(text).toContain("Python setup/build command");
      } finally {
        if (previous === undefined) {
          delete process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
        } else {
          process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = previous;
        }
      }
    });
  });

  it("does not block searching for package-install text under the SWE-bench guard", async () => {
    await withTempDir("openclaw-exec-env-guard-", async (tmp) => {
      const previous = process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
      process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = "1";
      try {
        const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });
        const result = await tool.execute("call-env-guard-search", {
          command: "printf '%s\\n' 'pip install -e .'",
          workdir: tmp,
        });
        const text = result.content.find((block) => block.type === "text")?.text ?? "";

        expect(result.details.status).toBe("completed");
        expect(text).toContain("pip install -e .");
      } finally {
        if (previous === undefined) {
          delete process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD;
        } else {
          process.env.OPENCLAW_SWEBENCH_EXEC_ENV_GUARD = previous;
        }
      }
    });
  });

  it("blocks shell env var injection tokens in python scripts before execution", async () => {
    await withTempDir("openclaw-exec-preflight-", async (tmp) => {
      const pyPath = path.join(tmp, "bad.py");

      await fs.writeFile(
        pyPath,
        [
          "import json",
          "# model accidentally wrote shell syntax:",
          "payload = $DM_JSON",
          "print(payload)",
        ].join("\n"),
        "utf-8",
      );

      const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });

      await expect(
        tool.execute("call1", {
          command: "python bad.py",
          workdir: tmp,
        }),
      ).rejects.toThrow(/exec preflight: detected likely shell variable injection \(\$DM_JSON\)/);
    });
  });

  it("blocks obvious shell-as-js output before node execution", async () => {
    await withTempDir("openclaw-exec-preflight-", async (tmp) => {
      const jsPath = path.join(tmp, "bad.js");

      await fs.writeFile(
        jsPath,
        ['NODE "$TMPDIR/hot.json"', "console.log('hi')"].join("\n"),
        "utf-8",
      );

      const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });

      await expect(
        tool.execute("call1", {
          command: "node bad.js",
          workdir: tmp,
        }),
      ).rejects.toThrow(
        /exec preflight: (detected likely shell variable injection|JS file starts with shell syntax)/,
      );
    });
  });

  it("skips preflight when script token is quoted and unresolved by fast parser", async () => {
    await withTempDir("openclaw-exec-preflight-", async (tmp) => {
      const jsPath = path.join(tmp, "bad.js");
      await fs.writeFile(jsPath, "const value = $DM_JSON;", "utf-8");

      const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });
      const result = await tool.execute("call-quoted", {
        command: 'node "bad.js"',
        workdir: tmp,
      });
      const text = result.content.find((block) => block.type === "text")?.text ?? "";
      expect(text).not.toMatch(/exec preflight:/);
    });
  });

  it("skips preflight file reads for script paths outside the workdir", async () => {
    await withTempDir("openclaw-exec-preflight-parent-", async (parent) => {
      const outsidePath = path.join(parent, "outside.js");
      const workdir = path.join(parent, "workdir");
      await fs.mkdir(workdir, { recursive: true });
      await fs.writeFile(outsidePath, "const value = $DM_JSON;", "utf-8");

      const tool = createExecTool({ host: "gateway", security: "full", ask: "off" });

      const result = await tool.execute("call-outside", {
        command: "node ../outside.js",
        workdir,
      });
      const text = result.content.find((block) => block.type === "text")?.text ?? "";
      expect(text).not.toMatch(/exec preflight:/);
    });
  });
});
