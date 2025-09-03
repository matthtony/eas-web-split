// Cloudflare Pages Function: POST /api/chat-stream (streams tokens)

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
	const indexUrl = new URL("/data/kb_index.json", request.url);
	const idxRes = await fetch(indexUrl.toString(), { cf: { cacheEverything: true } });
	if (!idxRes.ok) throw new Error("KB index not found at /data/kb_index.json");
	const index = await idxRes.json();
	const { shards = [] } = index || {};

	const chunks = [];
	const embeddings = [];
	const sources = [];
	for (const { file } of shards) {
		const shardUrl = new URL(`/data/${file}`, request.url);
		const sRes = await fetch(shardUrl.toString(), { cf: { cacheEverything: true } });
		if (!sRes.ok) throw new Error(`KB shard missing: ${file}`);
		const data = await sRes.json();
		if (Array.isArray(data.chunks)) chunks.push(...data.chunks);
		if (Array.isArray(data.embeddings)) embeddings.push(...(data.embeddings));
		if (Array.isArray(data.sources)) sources.push(...data.sources);
	}
	return { chunks, embeddings, sources };
}

export async function onRequestPost(context) {
	const { request, env } = context;
	try {
		const { message } = await request.json();
		if (!message) return new Response("Message is required", { status: 400 });

		const fallbackBase = "https://vercel-backend-3fpqg2iau-bans-projects-e190d146.vercel.app";
		const base = String(env.VERCEL_API_BASE || fallbackBase).replace(/\/$/, "");
		const headers = { "Content-Type": "application/json" };
		if (env.VERCEL_BYPASS_TOKEN) headers["x-vercel-protection-bypass"] = String(env.VERCEL_BYPASS_TOKEN);
		const upstream = await fetch(`${base}/api/chat-stream`, {
			method: "POST",
			headers,
			body: JSON.stringify({ message })
		});
		return new Response(upstream.body, {
			status: upstream.status,
			headers: { "Content-Type": "text/event-stream; charset=utf-8", "Cache-Control": "no-cache" }
		});
	} catch (e) {
		return new Response("Bad Request", { status: 400 });
	}
}



