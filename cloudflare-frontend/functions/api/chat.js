// Cloudflare Pages Function: POST /api/chat
// Uses precomputed embeddings from sharded files referenced by public/data/kb_index.json

function cosineSimilarity(a, b) {
	let dot = 0.0;
	let normA = 0.0;
	let normB = 0.0;
	for (let i = 0; i < a.length; i++) {
		dot += a[i] * b[i];
		normA += a[i] * a[i];
		normB += b[i] * b[i];
	}
	return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function tokenize(text) {
	return String(text)
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, " ")
		.split(" ")
		.filter(Boolean);
}

function keywordScore(queryTokens, text) {
	const hay = String(text).toLowerCase();
	let score = 0;
	for (const t of queryTokens) {
		if (hay.includes(t)) score += 1;
	}
	return score;
}

async function loadKb(request) {
	async function readJsonOrThrow(res, name) {
		const text = await res.text();
		try {
			return JSON.parse(text);
		} catch (e) {
			throw new Error(`${name} invalid JSON (status ${res.status}). Head: ${text.slice(0, 160)}`);
		}
	}

	const indexUrl = new URL("/data/kb_index.json", request.url);
	const idxRes = await fetch(indexUrl.toString(), { cf: { cacheEverything: true } });
	if (!idxRes.ok) throw new Error(`KB index not found at /data/kb_index.json (status ${idxRes.status})`);
	const index = await readJsonOrThrow(idxRes, "KB index");
	const { shards = [] } = index || {};

	const chunks = [];
	const embeddings = [];
	const sources = [];
	for (const { file } of shards) {
		const shardUrl = new URL(`/data/${file}`, request.url);
		const sRes = await fetch(shardUrl.toString(), { cf: { cacheEverything: true } });
		if (!sRes.ok) throw new Error(`KB shard missing: ${file} (status ${sRes.status})`);
		const data = await readJsonOrThrow(sRes, `KB shard ${file}`);
		if (Array.isArray(data.chunks)) chunks.push(...data.chunks);
		if (Array.isArray(data.embeddings)) embeddings.push(...(data.embeddings));
		if (Array.isArray(data.sources)) sources.push(...data.sources);
	}
	return { chunks, embeddings, sources };
}

export async function onRequestPost(context) {
	const { request, env } = context;
	try {
		const fallbackBase = "https://vercel-backend-1o2mfgxby-bans-projects-e190d146.vercel.app";
		const base = String(env.VERCEL_API_BASE || fallbackBase).replace(/\/$/, "");
		const headers = { "Content-Type": "application/json" };
		if (env.VERCEL_BYPASS_TOKEN) {
			headers["x-vercel-protection-bypass"] = String(env.VERCEL_BYPASS_TOKEN);
		}
		const upstream = await fetch(`${base}/api/chat`, {
			method: "POST",
			headers,
			body: await request.text()
		});
		const text = await upstream.text();
		return new Response(text, {
			status: upstream.status,
			headers: { "Content-Type": upstream.headers.get("content-type") || "application/json" }
		});
	} catch (e) {
		return new Response(JSON.stringify({ error: "Bad Request", detail: String(e) }), { status: 400 });
	}
}



