import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import crypto from "crypto";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const VERCEL_BACKEND_ROOT = path.join(__dirname, "..");
const FILES_DIR = (() => {
  const a = path.join(VERCEL_BACKEND_ROOT, "files");
  const b = path.join(VERCEL_BACKEND_ROOT, "files full");
  return fs.existsSync(a) ? a : b;
})();

const VB_PUBLIC_DATA_DIR = path.join(VERCEL_BACKEND_ROOT, "public", "data");
const VB_API_DIR = path.join(VERCEL_BACKEND_ROOT, "api");

// Also write to Cloudflare Pages public dir if present
const MONO_ROOT = path.join(VERCEL_BACKEND_ROOT, "..");
const CF_PUBLIC_DATA_DIR = path.join(MONO_ROOT, "cloudflare-frontend", "public", "data");

const SUPPORTED_EXTENSIONS = new Set([".pdf", ".txt", ".md"]);

function listSupportedFiles() {
  try {
    const entries = fs.readdirSync(FILES_DIR);
    return entries
      .map((name) => path.join(FILES_DIR, name))
      .filter((fullPath) => {
        try {
          const stat = fs.statSync(fullPath);
          const ext = path.extname(fullPath).toLowerCase();
          return stat.isFile() && SUPPORTED_EXTENSIONS.has(ext);
        } catch {
          return false;
        }
      });
  } catch {
    return [];
  }
}

function getCurrentSourceFileInfo(files) {
  return files.map((fullPath) => {
    const stat = fs.statSync(fullPath);
    return { name: path.basename(fullPath), size: stat.size, mtimeMs: stat.mtimeMs };
  });
}

async function main() {
  if (!fs.existsSync(FILES_DIR)) {
    console.error(`Files directory not found at ${FILES_DIR}`);
    process.exit(1);
  }
  const files = listSupportedFiles();
  if (!files.length) {
    console.error(`No supported files found in ${FILES_DIR}`);
    process.exit(1);
  }

  console.log(`Building RAW KB from ${files.length} file(s)…`);
  const rawFiles = [];
  for (const filePath of files) {
    const ext = path.extname(filePath).toLowerCase();
    const source = path.basename(filePath);
    const buf = fs.readFileSync(filePath);
    const bytes_b64 = buf.toString("base64");
    const sha256 = crypto.createHash("sha256").update(buf).digest("hex");
    const size = buf.length;
    let text = "";
    if (ext === ".pdf") {
      const parsed = await pdfParse(buf);
      text = parsed.text || "";
      rawFiles.push({ name: source, type: "pdf", size, sha256, bytes_b64, numPages: parsed?.numpages || null, text });
    } else {
      try { text = fs.readFileSync(filePath, "utf8"); } catch { text = ""; }
      rawFiles.push({ name: source, type: ext.slice(1), size, sha256, bytes_b64, text });
    }
  }

  const rawPayload = {
    version: 1,
    sourceFiles: getCurrentSourceFileInfo(files),
    files: rawFiles
  };

  // Ensure dirs
  for (const dir of [VB_PUBLIC_DATA_DIR, VB_API_DIR, CF_PUBLIC_DATA_DIR]) {
    try { fs.mkdirSync(dir, { recursive: true }); } catch {}
  }

  const targets = [
    path.join(VB_PUBLIC_DATA_DIR, "raw_kb.json"),
    path.join(VB_API_DIR, "raw_kb.json"),
    path.join(CF_PUBLIC_DATA_DIR, "raw_kb.json")
  ];
  for (const out of targets) {
    try {
      fs.writeFileSync(out, JSON.stringify(rawPayload));
      const sizeMb = (fs.statSync(out).size / (1024 * 1024)).toFixed(2);
      console.log(`Wrote ${out} (${sizeMb} MiB)`);
    } catch (e) {
      console.warn(`Failed writing ${out}:`, e?.message || e);
    }
  }
  console.log("RAW KB build complete.");
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


