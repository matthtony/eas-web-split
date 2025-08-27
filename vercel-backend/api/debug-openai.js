import OpenAI from "openai";

export default async function handler(req, res) {
  try {
    if (req.method !== "GET") {
      res.setHeader("Allow", "GET");
      return res.status(405).send("Method Not Allowed");
    }
    const apiKeyPresent = !!process.env.OPENAI_API_KEY;
    if (!apiKeyPresent) {
      return res.status(500).json({ ok: false, error: "OPENAI_API_KEY is not set" });
    }

    const base = process.env.OPENAI_API_BASE || "https://api.openai.com/v1";
    const r = await fetch(`${base}/embeddings`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ model: "text-embedding-3-small", input: "ping" })
    });
    const text = await r.text();
    if (!r.ok) {
      return res.status(500).json({ ok: false, error: `HTTP ${r.status}`, detail: text.slice(0, 500) });
    }
    let data;
    try { data = JSON.parse(text); } catch {}
    return res.json({ ok: true, dims: data?.data?.[0]?.embedding?.length || null });
  } catch (error) {
    const detail = error?.response?.data || error?.message || String(error);
    try {
      return res.status(500).json({ ok: false, error: "OpenAI call failed", detail });
    } catch {}
  }
}


