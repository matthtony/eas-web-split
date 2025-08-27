import OpenAI from "openai";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pdfParse = require("pdf-parse");

// -----------------------------------------------------------------------------
// Config
// -----------------------------------------------------------------------------
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const OPENAI_BASE_URL = process.env.OPENAI_API_BASE || "https://api.openai.com/v1";
const MAX_COMPLETION_TOKENS = Number(process.env.MAX_COMPLETION_TOKENS) || undefined;
const RETRIEVAL_K = Number(process.env.RETRIEVAL_K) || 8;
const MMR_LAMBDA = Number(process.env.MMR_LAMBDA) || 0.7; // tradeoff relevance vs diversity
const CONTEXT_CHAR_BUDGET = Number(process.env.CONTEXT_CHAR_BUDGET) || 120000; // ~30k tokens
const SCORE_THRESHOLD = Number(process.env.SCORE_THRESHOLD) || 0.22;

async function openaiPost(path, payload, options = {}) {
  const url = `${OPENAI_BASE_URL}${path}`;
  const controller = new AbortController();
  const timeout = options.timeoutMs ?? 20000;
  const to = setTimeout(() => controller.abort(), timeout);
  try {
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    if (!resp.ok) {
      const text = await resp.text();
      throw new Error(`OpenAI ${path} ${resp.status}: ${text.slice(0, 500)}`);
    }
    return await resp.json();
  } finally {
    clearTimeout(to);
  }
}

const CHUNK_SIZE = 2000;
const CHUNK_OVERLAP = 200;
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// Try both function-root and project-root layouts for Vercel vs local
const FILES_DIR_CANDIDATES = [
  path.join(__dirname, "files"),
  path.join(__dirname, "..", "files")
];
const FILES_FULL_DIR_CANDIDATES = [
  path.join(__dirname, "files full"),
  path.join(__dirname, "..", "files full")
];
const KB_CACHE_FILE_CANDIDATES = [
  // Prefer a file colocated with the function (produced by precompute script)
  path.join(__dirname, "kb_cache.json"),
  // Fallbacks for both function-root and project-root layouts
  path.join(__dirname, "data", "kb_cache.json"),
  path.join(__dirname, "..", "data", "kb_cache.json")
];
function pickExistingPath(candidates) {
  for (const p of candidates) {
    try {
      if (fs.existsSync(p)) return p;
    } catch {}
  }
  return null;
}
const DEFAULT_FILES_DIR = pickExistingPath(FILES_DIR_CANDIDATES) || FILES_DIR_CANDIDATES[0];
const ALT_FILES_DIR = pickExistingPath(FILES_FULL_DIR_CANDIDATES) || FILES_FULL_DIR_CANDIDATES[0];
const FILES_DIR = fs.existsSync(DEFAULT_FILES_DIR) ? DEFAULT_FILES_DIR : ALT_FILES_DIR;
// Keep a stable write target for local dev; Vercel writes may fail and are caught
const KB_CACHE_DIR = path.join(__dirname, "..", "data");
const KB_CACHE_FILE = path.join(KB_CACHE_DIR, "kb_cache.json");
const SUPPORTED_EXTENSIONS = new Set([".pdf", ".txt", ".md"]);

// -----------------------------------------------------------------------------
// In-memory cache so that subsequent Lambda invocations reuse the KB
// -----------------------------------------------------------------------------
let kbChunks = null;
let kbEmbeddings = null;
let kbChunkSources = null;

function chunkText(text, size = CHUNK_SIZE, overlap = CHUNK_OVERLAP) {
  const chunks = [];
  for (let i = 0; i < text.length; i += size - overlap) {
    chunks.push(text.slice(i, i + size));
    if (i + size >= text.length) break;
  }
  return chunks;
}

function cosineSimilarity(a, b) {
  let dot = 0.0,
    normA = 0.0,
    normB = 0.0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function generateQueryVariants(message, n = 4) {
  try {
    const resp = await openaiPost("/chat/completions", {
      model: "o4-mini",
      messages: [
        { role: "system", content: "Generate concise alternative phrasings of the user's question for retrieval. Return each variant on a new line. No numbering." },
        { role: "user", content: message }
      ],
      max_completion_tokens: 256
    }, { timeoutMs: 10000 });
    const text = resp?.choices?.[0]?.message?.content || "";
    const lines = text.split(/\n+/).map((s) => s.trim()).filter(Boolean);
    const uniq = Array.from(new Set(lines)).slice(0, n);
    return [message, ...uniq];
  } catch {
    return [message];
  }
}

function mmrSelect(scoredIdxs, embeddings, k, lambda = MMR_LAMBDA) {
  const selected = [];
  const remaining = new Set(scoredIdxs.map(({ idx }) => idx));
  // seed with best score
  selected.push(scoredIdxs[0].idx);
  remaining.delete(scoredIdxs[0].idx);
  while (selected.length < k && remaining.size) {
    let bestIdx = null;
    let bestScore = -Infinity;
    for (const idx of remaining) {
      const relevance = scoredIdxs.find((s) => s.idx === idx)?.score ?? 0;
      let diversity = 0;
      for (const s of selected) {
        const sim = cosineSimilarity(embeddings[idx], embeddings[s]);
        if (sim > diversity) diversity = sim;
      }
      const mmr = lambda * relevance - (1 - lambda) * diversity;
      if (mmr > bestScore) { bestScore = mmr; bestIdx = idx; }
    }
    if (bestIdx === null) break;
    selected.push(bestIdx);
    remaining.delete(bestIdx);
  }
  return selected;
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
    return {
      name: path.basename(fullPath),
      size: stat.size,
      mtimeMs: stat.mtimeMs
    };
  });
}

async function tryLoadKbFromCache() {
  try {
    const cachePath = pickExistingPath(KB_CACHE_FILE_CANDIDATES);
    if (!cachePath) return false;
    const raw = fs.readFileSync(cachePath, "utf8");
    const cached = JSON.parse(raw);

    if (!Array.isArray(cached.chunks) || !Array.isArray(cached.embeddings) || !Array.isArray(cached.sources)) {
      return false;
    }
    if (cached.model !== "text-embedding-3-small" || cached.chunkSize !== CHUNK_SIZE || cached.chunkOverlap !== CHUNK_OVERLAP) {
      return false;
    }

    const isVercel = !!process.env.VERCEL || !!process.env.NOW_REGION;
    // On Vercel, accept precomputed cache without validating mtimes (bundled files have different timestamps)
    if (!isVercel && fs.existsSync(FILES_DIR)) {
      const files = listSupportedFiles();
      if (files.length) {
        const currentInfo = getCurrentSourceFileInfo(files);
        const cachedInfo = cached.sourceFiles || [];
        if (currentInfo.length !== cachedInfo.length) return false;
        const byName = new Map(cachedInfo.map((i) => [i.name, i]));
        for (const info of currentInfo) {
          const c = byName.get(info.name);
          // Compare only by name and size to tolerate timestamp changes across environments
          if (!c || c.size !== info.size) return false;
        }
      }
    }

    kbChunks = cached.chunks;
    kbEmbeddings = cached.embeddings;
    kbChunkSources = cached.sources;
    console.log(`⚡ Loaded knowledge base from cache (${kbChunks.length} chunks).`);
    return true;
  } catch (e) {
    console.warn("Failed to load KB cache, will rebuild.", e?.message || e);
    return false;
  }
}

function saveKbToCache(sourceFilesInfo) {
  try {
    if (!fs.existsSync(KB_CACHE_DIR)) {
      fs.mkdirSync(KB_CACHE_DIR, { recursive: true });
    }
    const payload = {
      version: 1,
      model: "text-embedding-3-small",
      chunkSize: CHUNK_SIZE,
      chunkOverlap: CHUNK_OVERLAP,
      sourceFiles: sourceFilesInfo,
      chunks: kbChunks,
      embeddings: kbEmbeddings,
      sources: kbChunkSources
    };
    fs.writeFileSync(KB_CACHE_FILE, JSON.stringify(payload));
    console.log(`💾 Saved knowledge base cache to ${KB_CACHE_FILE}.`);
  } catch (e) {
    console.warn("Failed to write KB cache.", e?.message || e);
  }
}

async function buildKnowledgeBase() {
  if (kbChunks && kbEmbeddings && kbChunkSources) return; // already built in this runtime

  // Try to load from cache first (works even if files dir is missing in serverless)
  if (await tryLoadKbFromCache()) {
    return;
  }

  if (!fs.existsSync(FILES_DIR)) {
    console.warn(`⛔ Files directory not found at ${FILES_DIR}. Skipping KB build.`);
    kbChunks = [];
    kbEmbeddings = [];
    kbChunkSources = [];
    return;
  }

  const files = listSupportedFiles();

  if (!files.length) {
    console.warn(`⛔ No supported files found in ${FILES_DIR}. Skipping KB build.`);
    kbChunks = [];
    kbEmbeddings = [];
    kbChunkSources = [];
    return;
  }

  console.log(`📚 Building knowledge base from ${files.length} file(s)… this may take a moment…`);
  kbChunks = [];
  kbEmbeddings = [];
  kbChunkSources = [];

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
      const resp = await openai.embeddings.create({
        model: "text-embedding-3-small",
        input: chunk
      });
        kbEmbeddings.push(resp.data[0].embedding);
    }
  }
  console.log(`✅ Knowledge base ready with ${kbChunks.length} chunks from ${files.length} file(s).`);
  const sourceFilesInfo = getCurrentSourceFileInfo(files);
  saveKbToCache(sourceFilesInfo);
}

// -----------------------------------------------------------------------------
// Vercel Function handler
// -----------------------------------------------------------------------------
export default async function handler(req, res) {
  // CORS
  res.setHeader("Access-Control-Allow-Origin", process.env.CORS_ORIGIN || "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type, Authorization");
  if (req.method === "OPTIONS") {
    return res.status(204).end();
  }
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).send("Method Not Allowed");
  }

  // Read JSON body robustly (Vercel may not populate req.body automatically)
  let body = req.body;
  if (typeof body === "undefined") {
    const raw = await new Promise((resolve, reject) => {
      try {
        let data = "";
        req.on("data", (c) => (data += c));
        req.on("end", () => resolve(data));
        req.on("error", reject);
      } catch (e) {
        resolve("");
      }
    });
    try { body = raw ? JSON.parse(raw) : {}; } catch { body = {}; }
  } else if (typeof body === "string") {
    try { body = JSON.parse(body || "{}"); } catch {}
  }
  const { message } = body || {};
  if (!message) {
    return res.status(400).json({ error: "Message is required" });
  }

  try {
    await buildKnowledgeBase();

    // Multi-query retrieval; include as many chunks as fit the context budget
    let contextText = "";
    let maxScoreObserved = -Infinity;
    if (kbEmbeddings.length) {
      const variants = await generateQueryVariants(message, 4);
      const embResp = await Promise.all(variants.map((v) => openaiPost("/embeddings", { model: "text-embedding-3-small", input: v })));
      const queryEmbeddings = embResp.map((r) => r?.data?.[0]?.embedding || []);
      const scores = kbEmbeddings.map((emb, idx) => {
        let best = -Infinity;
        for (const q of queryEmbeddings) {
          const s = cosineSimilarity(q, emb);
          if (s > best) best = s;
        }
        return { idx, score: best };
      });
      scores.sort((a, b) => b.score - a.score);
      if (scores.length) maxScoreObserved = scores[0].score;
      let used = 0;
      const parts = [];
      for (const { idx } of scores) {
        const piece = `Source: ${kbChunkSources[idx]}\n${kbChunks[idx]}`;
        const extra = parts.length ? 5 : 0; // for separator length approx
        if (used + piece.length + extra > CONTEXT_CHAR_BUDGET) break;
        parts.push(piece);
        used += piece.length + extra;
      }
      contextText = parts.join("\n---\n");
    }

    const useInference = !contextText || maxScoreObserved < SCORE_THRESHOLD;
    const systemStrict = contextText
      ? `-Respond as complete and concise as possible, make sure the information given is accurate. \n-Do not answer questions outside of the knowledge files. \n-For each response, give source, reference, and page number at the end of each response for each information mentioned (reference to the documents within the documents because each file uploaded may contain multiple documents).\n-Crosscheck all of the information in the response with the reference. \n-Give the short conclusion first and follow with the explanation\n-Crosscheck and validate all responses strictly against the uploaded document sources before replying. Do not provide any response unless it can be fully supported with evidence from the documents.\n-If the sources state a numeric rule/ratio (e.g., '1 A requires 1 B'), USE that rule to compute implied quantities for the user’s asked amount using basic arithmetic. Show the calculation steps and cite the rule’s source. Do not invent rules.\n\n${contextText}`
      : "You are a helpful assistant. Answer concisely.";
    const systemInfer = contextText
      ? `-When sources are insufficient to fully answer, provide the best-effort inferred answer using clear assumptions and basic arithmetic/logic.\n-Label it as 'Best-effort inference' and prefer any partial evidence available.\n-If a numeric rule/ratio exists, scale it to the asked quantity and show steps.\n-If later documents contradict assumptions, state that the document rule should prevail.\n\n${contextText}`
      : "-No direct document evidence found. Provide a best-effort inferred answer using clear assumptions and basic arithmetic/logic. Keep it concise and label as 'Best-effort inference'. Ask for missing details if necessary.";

    const completionPayload = {
      model: "o3",
      messages: [
        {
          role: "system",
          content: useInference ? systemInfer : systemStrict
        },
        { role: "user", content: message }
      ]
    };
    if (MAX_COMPLETION_TOKENS) completionPayload.max_completion_tokens = MAX_COMPLETION_TOKENS;
    const completion = await openaiPost("/chat/completions", completionPayload, { timeoutMs: 30000 });
    const reply = completion?.choices?.[0]?.message?.content || "";
    res.json({ reply });
  } catch (error) {
    console.error(error);
    const detail = error?.response?.data || error?.message || String(error);
    res.status(500).json({ error: "Failed to get response from OpenAI", detail });
  }
} 