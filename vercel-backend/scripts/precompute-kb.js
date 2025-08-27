import OpenAI from "openai";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.join(__dirname, "..");
const DEFAULT_FILES_DIR = path.join(PROJECT_ROOT, "files");
const ALT_FILES_DIR = path.join(PROJECT_ROOT, "files full");
const FILES_DIR = fs.existsSync(DEFAULT_FILES_DIR) ? DEFAULT_FILES_DIR : ALT_FILES_DIR;
const KB_CACHE_DIR = path.join(PROJECT_ROOT, "data");
const KB_CACHE_FILE = path.join(KB_CACHE_DIR, "kb_cache.json");
const PUBLIC_DATA_DIR = path.join(PROJECT_ROOT, "public", "data");
const PUBLIC_KB_CACHE_FILE = path.join(PUBLIC_DATA_DIR, "kb_cache.json");
const PUBLIC_KB_INDEX_FILE = path.join(PUBLIC_DATA_DIR, "kb_index.json");
const SHARD_FILE_PREFIX = "kb_shard_";
const MAX_CHUNKS_PER_SHARD = 500; // keep each shard well under Pages' 25 MiB limit
// Also emit a colocated copy for the API function so Vercel bundles it next to the handler
const API_FUNC_CACHE_FILE = path.join(PROJECT_ROOT, "api", "kb_cache.json");

const CHUNK_SIZE = 2000; // larger chunks → fewer chunks → smaller JSON
const CHUNK_OVERLAP = 200;
const SUPPORTED_EXTENSIONS = new Set([".pdf", ".txt", ".md"]);

function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    const chunk = text.slice(i, i + size);
    chunks.push(chunk);
    if (i + size >= text.length) break;
  }
  return chunks;
}

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
  if (!process.env.OPENAI_API_KEY) {
    console.error("OPENAI_API_KEY is not set.");
    process.exit(1);
  }

  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  if (!fs.existsSync(FILES_DIR)) {
    console.error(`Files directory not found at ${FILES_DIR}`);
    process.exit(1);
  }

  const files = listSupportedFiles();
  if (!files.length) {
    console.error(`No supported files found in ${FILES_DIR}`);
    process.exit(1);
  }

  console.log(`Precomputing embeddings from ${files.length} file(s)…`);
  const kbChunks = [];
  const kbEmbeddings = [];
  const kbChunkSources = [];

  for (const filePath of files) {
    const ext = path.extname(filePath).toLowerCase();
    const source = path.basename(filePath);
    let text = "";

    if (ext === ".pdf") {
      const pdfBuffer = fs.readFileSync(filePath);
      const parsed = await pdfParse(pdfBuffer);
      text = parsed.text || "";
    } else {
      text = fs.readFileSync(filePath, "utf8");
    }

    const chunks = chunkText(text);
    for (const chunk of chunks) {
      kbChunks.push(chunk);
      kbChunkSources.push(source);
      const resp = await openai.embeddings.create({ model: "text-embedding-3-small", input: chunk });
      // Store full-precision embeddings for maximum accuracy (larger kb_cache.json)
      kbEmbeddings.push(resp.data[0].embedding);
    }
  }

  if (!fs.existsSync(KB_CACHE_DIR)) {
    fs.mkdirSync(KB_CACHE_DIR, { recursive: true });
  }

  const payload = {
    version: 1,
    model: "text-embedding-3-small",
    chunkSize: CHUNK_SIZE,
    chunkOverlap: CHUNK_OVERLAP,
    sourceFiles: getCurrentSourceFileInfo(files),
    chunks: kbChunks,
    embeddings: kbEmbeddings,
    sources: kbChunkSources
  };
  fs.writeFileSync(KB_CACHE_FILE, JSON.stringify(payload));
  console.log(`Wrote cache to ${KB_CACHE_FILE} with ${kbChunks.length} chunks.`);

  // Also emit sharded payload to public/data for Cloudflare Pages Functions static access
  try {
    if (!fs.existsSync(PUBLIC_DATA_DIR)) {
      fs.mkdirSync(PUBLIC_DATA_DIR, { recursive: true });
    }
    // Clean up old monolithic and shard files/index if present
    try {
      const existing = fs.readdirSync(PUBLIC_DATA_DIR);
      for (const name of existing) {
        if (
          name === path.basename(PUBLIC_KB_CACHE_FILE) ||
          name === path.basename(PUBLIC_KB_INDEX_FILE) ||
          name.startsWith(SHARD_FILE_PREFIX)
        ) {
          fs.unlinkSync(path.join(PUBLIC_DATA_DIR, name));
        }
      }
    } catch {}

    // Write sharded files + index to avoid the 25 MiB per-file limit on Pages
    const total = kbChunks.length;
    const shardsMeta = [];
    let shardIdx = 0;
    for (let start = 0; start < total; start += MAX_CHUNKS_PER_SHARD) {
      const end = Math.min(start + MAX_CHUNKS_PER_SHARD, total);
      const shard = {
        // keep shard payload minimal; index carries metadata
        chunks: kbChunks.slice(start, end),
        embeddings: kbEmbeddings.slice(start, end),
        sources: kbChunkSources.slice(start, end)
      };
      shardIdx += 1;
      const shardName = `${SHARD_FILE_PREFIX}${String(shardIdx).padStart(4, "0")}.json`;
      const shardPath = path.join(PUBLIC_DATA_DIR, shardName);
      fs.writeFileSync(shardPath, JSON.stringify(shard));
      const sizeMb = (fs.statSync(shardPath).size / (1024 * 1024)).toFixed(2);
      console.log(`Wrote shard ${shardName} with ${end - start} chunks (${sizeMb} MiB).`);
      shardsMeta.push({ file: shardName, chunkStart: start, chunkEnd: end });
    }

    const indexPayload = {
      version: 1,
      model: payload.model,
      chunkSize: payload.chunkSize,
      chunkOverlap: payload.chunkOverlap,
      sourceFiles: payload.sourceFiles,
      totalChunks: total,
      shards: shardsMeta
    };
    fs.writeFileSync(PUBLIC_KB_INDEX_FILE, JSON.stringify(indexPayload));
    console.log(`Wrote shard index to ${PUBLIC_KB_INDEX_FILE} with ${shardsMeta.length} shard(s).`);
  } catch (e) {
    console.warn(`Failed to write sharded KB to public/data:`, e?.message || e);
  }

  try {
    fs.writeFileSync(API_FUNC_CACHE_FILE, JSON.stringify(payload));
    console.log(`Wrote colocated cache for function to ${API_FUNC_CACHE_FILE}.`);
  } catch (e) {
    console.warn(`Failed to write colocated function cache at ${API_FUNC_CACHE_FILE}:`, e?.message || e);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});


