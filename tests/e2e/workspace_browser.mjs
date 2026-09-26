// harness/workspace.js DirWorkspace + codetools against a real FileSystemDirectoryHandle (an OPFS
// folder: the same API as a folder the user picks with showDirectoryPicker). No GPU needed.
//   NODE_PATH=... node tests/e2e/workspace_browser.mjs
import { loadPlaywright, chromiumPath, GPU_ARGS, serveRepo } from "./engine_synth.mjs";
const PORT = 18985;
async function pageMain() {
  const { DirWorkspace } = await import("/harness/workspace.js");
  const { codingTools } = await import("/harness/codetools.js");
  const out = [], res = [];
  const check = (n, ok, d = "") => { res.push(ok); out.push(`${ok ? "PASS" : "FAIL"} ${n}${d ? "  " + d : ""}`); };
  const root = await navigator.storage.getDirectory();
  await root.removeEntry("ws-test", { recursive: true }).catch(() => {});
  const dir = await root.getDirectoryHandle("ws-test", { create: true });
  const ws = new DirWorkspace(dir);
  await ws.write("src/add.js", "export const add = (a, b) => a - b;\n");
  await ws.write("node_modules/x/index.js", "return 1\n");
  await ws.write("README.md", "# t\n");
  const T = Object.fromEntries(codingTools(ws).map((t) => [t.name, t]));
  check("list_dir", (await T.list_dir.run({})) === "node_modules/\nREADME.md\nsrc/", JSON.stringify(await T.list_dir.run({})));
  check("read_file", (await T.read_file.run({ path: "src/add.js" })).includes("    1\texport const add"));
  check("search skips node_modules", (await T.search.run({ pattern: "return|a - b" })) === "src/add.js:1: export const add = (a, b) => a - b;");
  check("edit_file", (await T.edit_file.run({ path: "src/add.js", old_string: "a - b", new_string: "a + b" })).startsWith("edited") && (await ws.read("src/add.js")).includes("a + b"));
  check("write_file creates folders", (await T.write_file.run({ path: "test/add.test.js", content: "ok\n" })).startsWith("created") && await ws.exists("test/add.test.js"));
  let threw = false; try { await ws.read("../secret"); } catch { threw = true; }
  check("paths cannot leave the folder", threw);
  await root.removeEntry("ws-test", { recursive: true });
  return { out, ok: res.every(Boolean) };
}
const srv = serveRepo(PORT, {});
const { chromium } = await loadPlaywright();
const browser = await chromium.launch({ executablePath: chromiumPath(), args: GPU_ARGS });
const page = await browser.newPage();
await page.goto(`http://127.0.0.1:${PORT}/favicon.svg`);
const r = await page.evaluate(pageMain);
for (const l of r.out) console.log(l);
console.log(r.ok ? "WORKSPACE PASS" : "WORKSPACE FAIL");
await browser.close(); srv.close();
process.exit(r.ok ? 0 : 1);
