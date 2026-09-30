# Spill2Source: Oil Spill Detection, Drift Backtracking & Vessel Attribution (MVP)

**From ocean spill to its hidden source.**

A localhost demo that takes a SAR image of an oil slick and works backwards to a ranked list of candidate vessels that could have caused it.

```
SAR tile → U-Net segmentation → reverse-drift ensemble → AIS correlation → ranked leads → dashboard
 (Module 1)                       (Module 2)              (correlation)      (deck.gl)
```

**Demo video:** ` https://www.youtube.com/watch?v=LGT_BqLmPJA `

> This repository is the **MVP of the Satellite-First path** of the Spill2Source architecture. Vessels are presented as **investigative leads, not verdicts**. Nothing in the system or its API states or implies that a vessel discharged oil.

---

## Table of Contents

1. [MVP scope vs. full architecture](#1-mvp-scope-vs-full-architecture)
2. [How it works](#2-how-it-works)
3. [Key design decisions](#3-key-design-decisions)
4. [Quickstart](#4-quickstart)
5. [Using the dashboard](#5-using-the-dashboard)
6. [About the demo data](#6-about-the-demo-data)
7. [Vessel scoring](#7-vessel-scoring)
8. [API reference](#8-api-reference)
9. [Ports](#9-ports)
10. [Troubleshooting](#10-troubleshooting)
11. [Known limits](#11-known-limits)

---

## 1. MVP scope vs. full architecture

The full Spill2Source design is a dual-direction system: **Satellite-First** (spill → vessel) and **AIS-First** (vessel → spill). This repository implements the Satellite-First core end to end.

| Capability | MVP status |
|---|---|
| SAR oil-slick segmentation (U-Net, val IoU 0.651, ~86–87% pixel accuracy) | ✅ Implemented |
| Reverse-drift ensemble with Bayesian (ABC) posterior over release location and time | ✅ Implemented (custom Lagrangian integrator) |
| Forward trajectory forecast from the estimated release point | ✅ Implemented |
| AIS correlation and ranked candidate vessels | ✅ Implemented |
| Live dashboard (deck.gl map, Socket.io stage streaming) | ✅ Implemented |
| ResNet-18 gatekeeper classifier and DeepLabV3+ segmentation | 🔜 Planned |
| Cross-modal verification (temporal, multi-sensor, environmental, natural-seepage / existing-event checks) | 🔜 Planned |
| OpenDrift-based engine (Eulerian, Lagrangian, Adjoint), wind forcing, weathering | 🔜 Planned |
| **AIS-First path:** Isolation Forest anomaly detection, AIS-gap weighting, satellite tasking | 🔜 Planned |
| Streaming and scale-out stack (Kafka, Redis, PostGIS, MinIO) | 🔜 Planned |

The MVP stack is **FastAPI + Node.js + MongoDB + React/deck.gl**. The technical approach document describes the full production design.

---

## 2. How it works

Three FastAPI services, a Node.js orchestrator and a React dashboard.

```
                    ┌────────────────────────── Gateway (Node, :4000) ──────────────────────────┐
  Dashboard ──────► │  REST + Socket.io  ── orchestrates ──►  Detection   (FastAPI :8001)       │
  (React, :5173)    │                                         Backtrack   (FastAPI :8002)       │
                    │                                         AIS         (FastAPI :8003)       │
                    └──────────────────────────────┬────────────────────────────────────────────┘
                                                   ▼
                                            MongoDB (:27017)
```

| Stage | What it does |
|---|---|
| **1. Detection (Module 1)** | Runs the trained U-Net on the SAR tile and produces the slick mask, outline, area and centroid. |
| **2. Backtracking (Module 2)** | Seeds candidate release hypotheses (location × time), advects them backward through an ocean-current field, and scores each against the observed slick (centroid offset, area, elongation) to produce an origin posterior and uncertainty radius. |
| **3. AIS correlation** | Scores every vessel by how close it was to the posterior, and when, within the estimated release window. |
| **4. Forecast** | Advects the estimated release point forward for a chosen duration. |

### Repository layout

```
module1_detection/     U-Net inference, pre/post-processing, PNG demo entry point
module2_backtracking/  Advection, ABC backtracking, forward tracking, config
shared_apis/           Ocean-current forcing clients (Open-Meteo, gridded, synthetic fallback)
backend/               detection_service, backtrack_service, ais_service, gateway
frontend/              React + deck.gl dashboard
scripts/               Catalog builder, DB seeder, health check, Windows launchers
```

---

## 3. Key design decisions

Two integration problems only became visible once the models were wired together. Both would have let the demo look like it worked while producing meaningless numbers, so both are handled explicitly.

### 3a. A spatially uniform current field flattens the posterior

A single-point current query returns the same velocity everywhere. In uniform flow:

- a particle cloud advected backward *h* hours and then forward *h* hours returns exactly to its start for every *h*, so the centroid-offset term is ~0 for every candidate;
- with no velocity gradient, cloud area and elongation are conserved exactly, so those two terms are identical for every candidate as well.

All three likelihood terms then coincide, the posterior comes out flat, and the origin estimate degrades into an unweighted average along the drift track. Nothing crashes, but the result stops being Bayesian.

**Fix:** `OpenMeteoGriddedCurrentField` samples a 4×4 grid across the search box in one batched request and interpolates bilinearly. It uses the same data and API call count but gives the ABC kernel a real velocity gradient to discriminate on. Fallback order is gridded → single-point → synthetic. Set `OPENMETEO_GRID_SIDE=1` in `.env` to disable it.

### 3b. The release footprint must be sized from the observation

The backtracker scores candidates partly on `log(sim_area / obs_area)`. Advection is near area-preserving, so `sim_area` stays close to the seed circle's area. A default seed radius of 0.3 km against a ~30 km² observed slick (equivalent radius ~3.2 km) gives an area ratio near 0.01. The likelihood then underflows to zero for every candidate and the posterior collapses to uniform.

**Fix:** the backtrack service sizes the seed circle from the detection: `radius = sqrt(area_km² / π) × RELEASE_FOOTPRINT_FRAC`. Measured on one demo scene (`demo_img_01`):

| Seed radius | Effective sample size | Result |
|---|---|---|
| 0.30 km (module default) | 1.9 / 24 | All mass on one candidate; origin 35 km off |
| 3.19 km (fitted to observation) | 12.9 / 24 | Well-spread posterior, ±3.7 km |

At `RELEASE_FOOTPRINT_FRAC=1.0` the area term is neutral and discrimination comes from centroid offset and elongation, which is the honest position for an integrator that models no spreading or weathering.

### Self-test

```bash
python run_module2.py --self-test
# [self-test] most_likely_origin error vs. true origin: 0.18 km
# [self-test] true hour ranks #1/12
# [self-test] OK
```

On a synthetic case with a known release point, the backtracker recovers the origin to 0.18 km and ranks the true release hour first.

---

## 4. Quickstart

### 4.1 Prerequisites

| Tool | Version | Check |
|---|---|---|
| Python | 3.10–3.12 | `python --version` |
| Node.js | 18+ | `node --version` |
| MongoDB Community Server | 6.0+ | `mongod --version` |

The trained checkpoint `module1_detection/weights/unet_best.pt` (31 MB, epoch 12, val IoU 0.651) must be present.

### 4.2 MongoDB

There is no manual "create database" step. MongoDB creates databases and collections on first write, so starting the server and running the seed script is enough.

Install [MongoDB Community Server](https://www.mongodb.com/try/download/community) (Windows MSI: choose *Complete* and *Install MongoDB as a Service*), then confirm it is running from an Administrator prompt:

```bat
sc query MongoDB
net start MongoDB
```

To run it in the foreground instead:

```bat
mkdir C:\data\db
"C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" --dbpath C:\data\db
```

(Adjust `8.0` to your installed version.)

### 4.3 Python environment

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The CPU build of PyTorch is recommended because the U-Net does one 256×256 forward pass per scene and gains nothing from CUDA here. If `python` opens the Microsoft Store, disable the Python entries under *Settings → Apps → Advanced app settings → App execution aliases*.

### 4.4 Node dependencies

```bat
cd backend\gateway && npm install && cd ..\..
cd frontend && npm install && cd ..
```

Lockfiles are included, so you get the tested versions.

### 4.5 Configuration

```bat
copy .env.example .env
```

Every value in `.env.example` is already the default, so an unedited copy works. The free Open-Meteo Marine API needs **no account or key**, so leave `OPENMETEO_API_KEY` blank. Set it only if you use a commercial plan, which uses a separate paid host.

### 4.6 Build the catalog and seed the database

```bat
python scripts\make_demo_catalog.py
python scripts\seed_db.py
```

The first runs the U-Net on each demo tile and fits each scene's geographic bounding box (see [section 6](#6-about-the-demo-data)). The second creates the `oilspill` database and loads `demo_images` and `ships`. Re-running is safe. Add `--reset-events` to also clear previous pipeline runs.

### 4.7 Preflight check

```bat
python scripts\check_health.py
```

Checks packages, MongoDB and its collections, the U-Net checkpoint, and makes a **live call to Open-Meteo** to report whether you will get real current data or the synthetic fallback. Exits non-zero if anything needs fixing. Run this before every demo.

### 4.8 Run

```bat
scripts\start_all.bat
```

Five windows open. Wait about 15 seconds for the detection service to load the checkpoint, then open **http://localhost:5173**.

To start services manually, one terminal each from the project root:

```bat
python backend\detection_service\main.py       :8001
python backend\backtrack_service\main.py       :8002
python backend\ais_service\main.py             :8003
cd backend\gateway && node server.js           :4000
cd frontend && npm run dev                     :5173
```

`scripts\stop_all.bat` frees all five ports.

---

## 5. Using the dashboard

1. Pick a scene from the thumbnails.
2. Press **Run analysis**.
3. The map fills in stage by stage over Socket.io:
   - the detected slick outline (amber);
   - the candidate-origin cloud sweeping oldest to newest, settling into the full posterior with an uncertainty ring;
   - each vessel pulsing as it is checked, then the ranked panel.
4. Choose a duration and press **Forecast** to advect the estimated release point forward.
5. Click any posterior cell or vessel marker for its numbers. Click a vessel row to expand its scoring breakdown.

A full run takes roughly **5–8 seconds**: about 2.5 s for detection (first run includes model load), 1–3 s for the drift ensemble, and under a second for correlation.

### Reading the confidence badge

The badge reports the drift model's own diagnostics, with a specific message for each failure mode:

| Message | Meaning |
|---|---|
| **Not physically meaningful** | The current API was unreachable and the run used the synthetic gyre field. The pipeline is demonstrated but the origin is not a result. Fix connectivity and re-run. |
| **Ensemble collapsed** | Every candidate scored near zero and the posterior went uniform. The 1–48 h search window probably does not bracket the release. |
| **Origin uncertain** | Effective sample size is below 15% of the ensemble, so nearly all mass sits on one hypothesis. Treat `uncertainty_radius_km` with caution. |
| **Weak current gradient** | The field is nearly uniform across the search area, so release times are hard to tell apart. Space is better constrained than time. |

---

## 6. About the demo data

### SAR tiles

`scene_01_delta.png` is a single real SAR tile (VV panel), resized to 512×512. The other three scenes are geometric variants of it (rotate 90°, mirror, rotate 270° and crop). They keep the real SAR texture, so the network is still segmenting a genuine slick signature, but each gives a different mask geometry and therefore a different detection.

To use your own tiles, place PNGs in `data/demo_images/`, edit the `SCENES` list at the top of `scripts/make_demo_catalog.py`, then re-run it followed by `seed_db.py`.

### Why the bounding boxes are fitted

The tiles carry no CRS, geotransform or acquisition time, so each scene needs a bounding box and timestamp assigned by hand. Doing that manually is fiddly, so `make_demo_catalog.py` inverts it: you state where the slick should be, it runs the U-Net, finds the primary component's centroid, and solves for the box that puts it there. This keeps the detection, the drift posterior and the AIS data in one patch of ocean.

Each scene spans about 0.153° × 0.135° (~15 × 15 km at 28.4° N), giving detected slicks of a plausible 29–34 km².

### Geographic coherence

All scenes sit in the Mississippi Canyon area of the Gulf of Mexico, inside the seeded vessel field (7 vessels, tracks running 2026-09-03 16:12 Z to 2026-09-04 06:33 Z).

| Scene | Slick centre | Imaged |
|---|---|---|
| Mississippi Canyon | 28.310 N, 89.020 W | 06:12 Z |
| Mars Ridge | 28.455 N, 89.145 W | 05:30 Z |
| Shelf Edge | 28.235 N, 89.190 W | 04:45 Z |
| West Flank | 28.575 N, 89.255 W | 06:40 Z |

### Data disclaimer

**Scene locations and timestamps are assigned for demonstration, and the AIS tracks and vessel names are synthetic.** The demo shows that the pipeline works end to end. It is not a validated attribution of any real event or vessel.

---

## 7. Vessel scoring

For every reported position of a vessel inside `release_time_window`:

```
grid_proximity = Σ over cells c of origin_probability_grid:
                   p(c) · exp( −distance_km(vessel, c) / SIGMA_DIST_KM )
```

`SIGMA_DIST_KM` is imported from Module 2's `config.py` (default 8.0), so "close" means the same thing here as in the ABC kernel that produced the posterior.

Scoring against the **whole grid** rather than the point estimate matters: a vessel on a broad ridge of moderate probability is a better lead than one near the mean of a bimodal posterior, which may be water no candidate favoured.

The raw sum is normalised against what an ideal vessel sitting exactly on `most_likely_origin` would score, so the number reads as "how close to ideal is this position". Normalising against the best actual vessel instead would hand a top score to the least-bad candidate in a fleet that was all far away.

```
time_centrality  = 1 − 2·|t − window_centre| / window_width      (clamped to [0, 1])
suspicion_score  = 100 · (0.7 · normalised_proximity + 0.3 · time_centrality)
```

Vessels with no position inside the window are still returned, with `time_centrality = 0` and `in_release_window = false`, scored on their nearest in-time position. Showing that a vessel was checked, and why it ranks lower, is itself a result, and the map shows every vessel being considered.

### Illustrative output

```
 57.2  VESSEL-A   Other        3.7 km   outside window
 49.6  VESSEL-B   Passenger    5.9 km
 25.1  VESSEL-C   Other       27.8 km
 21.6  VESSEL-D   Tanker      23.3 km
 14.2  VESSEL-E   Towing      28.0 km
  4.5  VESSEL-F   Cargo       24.6 km   outside window
  2.4  VESSEL-G   Tanker      38.5 km
```

A vessel can rank first on proximity even when its nearest position falls outside the window, because proximity carries 70% of the score. The `outside window` flag is shown so an investigator can weigh that.

> **A high score means a vessel was in the right water at the right time according to a drift model with real uncertainty. It is an investigative lead, not evidence of discharge.** The API returns no field that phrases it otherwise.

---

## 8. API reference

Everything goes through the gateway on port 4000. Vite proxies `/api` and the socket handshake, so the frontend only talks to its own origin.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/scenes` | Catalog for the scene picker |
| GET | `/api/scenes/:id/image` | Tile PNG |
| POST | `/api/pipeline/run` | `{image_id}` → runs the whole chain |
| POST | `/api/drift/forward` | Forecast from an origin |
| GET | `/api/events/:id` | A previous run from MongoDB |
| GET | `/api/health` | Fan-out health check across all services |

**Socket.io events.** Each carries its payload, not just a name, so a dropped socket degrades to "the map fills in at the end" instead of an empty screen:

```
stage:started          {stage}
stage:detection_done   {detection}
stage:backtrack_done   {backtrack}
stage:ais_done         {ranked_vessels, ais_debug}
stage:forward_done     {forward}
stage:pipeline_done    {elapsed_seconds}
stage:error            {stage, message}
```

The Python services can be reached directly on 8001–8003 for isolated testing, and each serves interactive docs at `/docs`.

### MongoDB collections

| Collection | Contents |
|---|---|
| `demo_images` | Curated catalog: bbox, timestamp, provenance |
| `spill_events` | One document per run: detection, posterior, vessels, forecast |
| `ships` | Vessel tracks, plus a `2dsphere` index on `track_geometry` |

The AIS scorer walks paths in Python instead of querying by geometry, which is faster than a round trip at seven vessels. The index is there so a larger dataset can switch to a `$near` pre-filter without a schema change.

---

## 9. Ports

| Service | Port |
|---|---|
| Frontend (Vite) | 5173 |
| Gateway (Express + Socket.io) | 4000 |
| Detection (FastAPI) | 8001 |
| Backtrack (FastAPI) | 8002 |
| AIS (FastAPI) | 8003 |
| MongoDB | 27017 |

Change them in `.env`. `vite.config.js` also hardcodes the gateway port in its proxy target, so update that too if you move it.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot reach the gateway" | The gateway is not running, or Mongo was down when it started. Start `mongod`, then restart the gateway. |
| "The ships collection is empty" | Run `python scripts\seed_db.py`. |
| Badge says "Not physically meaningful" | Open-Meteo was unreachable. Run `python scripts\check_health.py`, which tests the API directly. Corporate networks and VPNs sometimes block it. |
| Map is blank but panels have data | The CARTO basemap needs internet. The app falls back to a flat ocean fill and deck.gl layers still render. |
| Detection returns 422 "no slick above threshold" | Lower the threshold with `DETECT_THRESHOLD=0.35` in `.env`, or pick another scene. |
| Port already in use | Run `scripts\stop_all.bat`. |
| Torch install is very large | Use the CPU index URL from section 4.3. |
| Detection service takes ~15 s to start | Expected: it is loading the 31 MB checkpoint, and it is warmed at startup so the first request does not pay for it. |

---

## 11. Known limits

This is an MVP, and these limits are deliberate scope choices.

- **Wind is ignored.** `wind_u_mps` / `wind_v_mps` default to 0 and no wind API is called. Real leeway on a surface slick is roughly 2–4% of wind speed and is not modelled.
- **The current grid is coarse.** Four points across a ~1.6° box resolves large-scale shear, not mesoscale eddies. A CMEMS NetCDF through `NetCDFGridField` is the higher-fidelity path: point `CURRENT_CACHE_PATH` at one if you have it.
- **No diffusion, spreading or weathering.** Advection is deterministic, so patch area is nearly conserved. This is why the release footprint is sized from the observation (section 3b), and why forecast spread is smaller than real forecast uncertainty.
- **Demo tiles are single-polarisation.** VV is duplicated into the VH channel because the PNGs have one band. The network was trained on dual-pol input, so it has less to work with than on an operational scene.
- **AIS data is synthetic**, and vessel names are fictional.
- **Bounding boxes are assigned, not measured** (see section 6).
- **Satellite-First only.** The AIS-First path, AIS-gap handling, verification layer and production data stack are part of the full architecture and are not implemented here (see section 1).

---

## Disclaimer

Spill2Source outputs are decision-support leads for human investigators. They are not evidence of discharge and do not identify a responsible party.