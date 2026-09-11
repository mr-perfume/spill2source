/**
 * Every call goes through the gateway on the same origin (Vite proxies /api
 * in development), so the frontend never needs to know which Python service
 * answered.
 */
const BASE = "/api";

async function request(path, options) {
  const res = await fetch(`${BASE}${path}`, options);
  const text = await res.text();
  let body;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    body = { error: text.slice(0, 300) };
  }
  if (!res.ok) {
    throw new Error(body.error || `${res.status} ${res.statusText}`);
  }
  return body;
}

export const fetchScenes = () => request("/scenes");

export const fetchHealth = () => request("/health").catch((e) => ({ status: "unreachable", error: e.message }));

export const runPipeline = (imageId) =>
  request("/pipeline/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: imageId }),
  });

export const runForward = (payload) =>
  request("/drift/forward", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

export const sceneImageUrl = (imageId) => `${BASE}/scenes/${encodeURIComponent(imageId)}/image`;
