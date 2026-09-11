/**
 * gateway/server.js
 * ---------------------------------------------------------------------------
 * The orchestrator. It owns the pipeline sequence, writes each stage's result
 * onto one spill_events document, and emits a Socket.io event as each stage
 * lands so the map can reveal a layer at a time instead of appearing all at
 * once when the last service returns.
 *
 * The stage events carry their payload, not just a name. The frontend then
 * renders from the socket message and treats the final HTTP response as a
 * consistency check rather than the thing it was waiting for -- which means a
 * dropped socket degrades into "the map fills in at the end" instead of an
 * empty screen.
 *
 * Routes:
 *   POST /api/pipeline/run       {image_id}  -> full assembled result
 *   POST /api/drift/forward      forecast from an origin (Section 6b)
 *   GET  /api/scenes             proxied catalog for the scene picker
 *   GET  /api/scenes/:id/image   proxied tile
 *   GET  /api/events/:id         a previous run
 *   GET  /api/health             fan-out health check across all services
 */

const express = require("express");
const http = require("http");
const cors = require("cors");
const { Server } = require("socket.io");
const { MongoClient } = require("mongodb");
const path = require("path");

require("dotenv").config({ path: path.join(__dirname, "..", "..", ".env") });

const PORT = parseInt(process.env.GATEWAY_PORT || "4000", 10);
const DETECTION_URL = process.env.DETECTION_URL || "http://127.0.0.1:8001";
const BACKTRACK_URL = process.env.BACKTRACK_URL || "http://127.0.0.1:8002";
const AIS_URL = process.env.AIS_URL || "http://127.0.0.1:8003";
const MONGODB_URI = process.env.MONGODB_URI || "mongodb://localhost:27017";
const MONGODB_DB = process.env.MONGODB_DB || "oilspill";

const app = express();
app.use(cors());
app.use(express.json({ limit: "12mb" }));

const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

let db = null;
MongoClient.connect(MONGODB_URI, { serverSelectionTimeoutMS: 5000 })
  .then((client) => {
    db = client.db(MONGODB_DB);
    console.log(`[gateway] mongo connected: ${MONGODB_DB}`);
  })
  .catch((err) => {
    console.error(`[gateway] mongo unavailable at ${MONGODB_URI}: ${err.message}`);
    console.error("[gateway] start mongod, then restart this service");
  });

/* -------------------------------------------------------------------------
 * Service calls
 * ---------------------------------------------------------------------- */

/**
 * Calls a downstream FastAPI service and turns its error shape into
 * something the frontend can display. FastAPI puts the useful message in
 * `detail`, which a bare `res.statusText` would throw away.
 */
async function callService(url, body, stageName, timeoutMs = 180000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let res;
  try {
    res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") {
      throw new StageError(stageName, 504, `${stageName} did not respond within ${timeoutMs / 1000}s`);
    }
    throw new StageError(
      stageName,
      503,
      `Cannot reach the ${stageName} service at ${url}. Is it running? (${err.message})`
    );
  }
  clearTimeout(timer);

  const text = await res.text();
  let payload;
  try {
    payload = JSON.parse(text);
  } catch {
    payload = { detail: text.slice(0, 400) };
  }
  if (!res.ok) {
    const detail = payload?.detail || res.statusText;
    throw new StageError(stageName, res.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

class StageError extends Error {
  constructor(stage, status, message) {
    super(message);
    this.stage = stage;
    this.status = status;
  }
}

/* -------------------------------------------------------------------------
 * Pipeline
 * ---------------------------------------------------------------------- */
app.post("/api/pipeline/run", async (req, res) => {
  const { image_id: imageId, run_forward: runForward, forward_hours: forwardHours } = req.body || {};
  if (!imageId) {
    return res.status(400).json({ error: "image_id is required" });
  }

  const runId = `run_${Date.now().toString(36)}`;
  const started = Date.now();
  const emit = (event, payload) => io.emit(event, { run_id: runId, image_id: imageId, ...payload });

  emit("stage:started", { stage: "detection" });

  try {
    /* -- 1. detection ---------------------------------------------------- */
    const detection = await callService(`${DETECTION_URL}/detect`, { image_id: imageId }, "detection");
    emit("stage:detection_done", { detection });

    /* -- 2. drift backtrack ---------------------------------------------- */
    emit("stage:started", { stage: "backtrack", detection_id: detection.detection_id });
    const backtrack = await callService(
      `${BACKTRACK_URL}/drift/backtrack`,
      {
        detection_id: detection.detection_id,
        lat: detection.lat,
        lon: detection.lon,
        polygon_geojson: detection.polygon_geojson,
        timestamp_utc: detection.timestamp_utc,
        area_km2: detection.area_km2,
      },
      "backtrack"
    );
    emit("stage:backtrack_done", { detection_id: detection.detection_id, backtrack });

    /* -- 3. vessel correlation ------------------------------------------- */
    emit("stage:started", { stage: "ais", detection_id: detection.detection_id });
    const ais = await callService(
      `${AIS_URL}/ais/correlate`,
      {
        detection_id: detection.detection_id,
        origin_probability_grid: backtrack.origin_probability_grid,
        most_likely_origin: backtrack.most_likely_origin,
        release_time_window: backtrack.release_time_window,
      },
      "ais"
    );
    emit("stage:ais_done", {
      detection_id: detection.detection_id,
      ranked_vessels: ais.ranked_vessels,
      ais_debug: ais._debug,
    });

    /* -- 4. optional forecast -------------------------------------------- */
    let forward = null;
    if (runForward) {
      emit("stage:started", { stage: "forward", detection_id: detection.detection_id });
      forward = await callService(
        `${BACKTRACK_URL}/drift/forward`,
        {
          detection_id: detection.detection_id,
          lat: backtrack.most_likely_origin.lat,
          lon: backtrack.most_likely_origin.lon,
          start_time_utc: backtrack.most_likely_origin.time,
          duration_hours: forwardHours || 12,
          area_km2: detection.area_km2,
        },
        "forward"
      );
      emit("stage:forward_done", { detection_id: detection.detection_id, forward });
    }

    const result = {
      run_id: runId,
      detection_id: detection.detection_id,
      image_id: imageId,
      detection,
      backtrack,
      ranked_vessels: ais.ranked_vessels,
      ais_debug: ais._debug,
      forward,
      elapsed_seconds: Number(((Date.now() - started) / 1000).toFixed(2)),
    };

    if (db) {
      await db.collection("spill_events").updateOne(
        { _id: detection.detection_id },
        {
          $set: {
            run_id: runId,
            origin_probability_grid: backtrack.origin_probability_grid,
            most_likely_origin: backtrack.most_likely_origin,
            release_time_window: backtrack.release_time_window,
            uncertainty_radius_km: backtrack.uncertainty_radius_km,
            backtrack_debug: backtrack._debug,
            ranked_vessels: ais.ranked_vessels,
            forward,
            completed_at: new Date(),
          },
        },
        { upsert: true }
      );
    }

    emit("stage:pipeline_done", { detection_id: detection.detection_id, elapsed_seconds: result.elapsed_seconds });
    res.json(result);
  } catch (err) {
    const stage = err.stage || "pipeline";
    console.error(`[gateway] ${stage} failed: ${err.message}`);
    emit("stage:error", { stage, message: err.message });
    res.status(err.status || 500).json({ error: err.message, stage });
  }
});

/* -------------------------------------------------------------------------
 * Standalone forecast, for the duration control on an already-finished run
 * ---------------------------------------------------------------------- */
app.post("/api/drift/forward", async (req, res) => {
  try {
    const forward = await callService(`${BACKTRACK_URL}/drift/forward`, req.body, "forward");
    io.emit("stage:forward_done", { detection_id: req.body.detection_id, forward });
    if (db && req.body.detection_id) {
      await db
        .collection("spill_events")
        .updateOne({ _id: req.body.detection_id }, { $set: { forward } });
    }
    res.json(forward);
  } catch (err) {
    res.status(err.status || 500).json({ error: err.message, stage: err.stage || "forward" });
  }
});

/* -------------------------------------------------------------------------
 * Catalog proxying, so the frontend only ever talks to one origin
 * ---------------------------------------------------------------------- */
app.get("/api/scenes", async (_req, res) => {
  try {
    const r = await fetch(`${DETECTION_URL}/scenes`);
    if (!r.ok) throw new Error(`detection service returned ${r.status}`);
    res.json(await r.json());
  } catch (err) {
    res.status(503).json({
      error: `Cannot load the scene catalog from the detection service: ${err.message}`,
    });
  }
});

app.get("/api/scenes/:id/image", async (req, res) => {
  try {
    const r = await fetch(`${DETECTION_URL}/scenes/${encodeURIComponent(req.params.id)}/image`);
    if (!r.ok) return res.status(r.status).end();
    res.set("Content-Type", "image/png");
    res.set("Cache-Control", "public, max-age=3600");
    res.send(Buffer.from(await r.arrayBuffer()));
  } catch (err) {
    res.status(503).json({ error: err.message });
  }
});

app.get("/api/events/:id", async (req, res) => {
  if (!db) return res.status(503).json({ error: "Database not connected" });
  const doc = await db.collection("spill_events").findOne({ _id: req.params.id });
  if (!doc) return res.status(404).json({ error: `No run ${req.params.id}` });
  res.json(doc);
});

app.get("/api/health", async (_req, res) => {
  const probe = async (name, url) => {
    try {
      const r = await fetch(`${url}/health`, { signal: AbortSignal.timeout(4000) });
      return { name, reachable: r.ok, ...(await r.json()) };
    } catch (err) {
      return { name, reachable: false, error: err.message };
    }
  };
  const services = await Promise.all([
    probe("detection", DETECTION_URL),
    probe("backtrack", BACKTRACK_URL),
    probe("ais", AIS_URL),
  ]);
  const allUp = services.every((s) => s.reachable) && !!db;
  res.status(allUp ? 200 : 503).json({ status: allUp ? "ok" : "degraded", mongo: !!db, services });
});

io.on("connection", (socket) => {
  console.log(`[gateway] client connected: ${socket.id}`);
  socket.on("disconnect", () => console.log(`[gateway] client gone: ${socket.id}`));
});

server.listen(PORT, () => {
  console.log(`[gateway] listening on http://127.0.0.1:${PORT}`);
  console.log(`[gateway] detection=${DETECTION_URL} backtrack=${BACKTRACK_URL} ais=${AIS_URL}`);
});
