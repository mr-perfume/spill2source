# Oil Spill Detection, Drift & Vessel Attribution — MVP

A localhost demo that takes a SAR image of an oil slick and works backwards to a
ranked list of vessels that could have caused it.

```
SAR tile  →  U-Net segmentation  →  reverse drift ensemble  →  AIS correlation  →  ranked leads
 (Module 1)                          (Module 2)                (new logic)         (dashboard)
```

Both models were already trained. This repository is the integration: three
FastAPI wrappers, a Node orchestrator, MongoDB, and a deck.gl dashboard.

---

## 1. Read this first: what changed in your uploaded code

The model code arrived with markdown-mangled source. Every double underscore
and every `**` exponent had been eaten, so **none of these files would import**.
They are fixed in this bundle:

| File | Was | Now |
|---|---|---|
| `module2_backtracking/*.py` (all four) | `from _future_ import annotations` | `from __future__ import annotations` |
| `shared_apis/drift_forcing_client.py` | same, plus `def _init_(` in four classes | `def __init__(` |
| `module1_detection/postprocessing.py` | same, plus `if __name__ == "_main_"` | `"__main__"` |
| five call sites | `type(e)._name_` | `type(e).__name__` |
| `run_module2.py:106` | `out["debug"]["current_source"] = type(...).name_` | `out["_debug"][...] = type(...).__name__` |
| `drift_forcing_client.py:407` | `np.exp(-(r_km * 2) / (2 * length_scale_km * 2))` | `... r_km ** 2 ... length_scale_km ** 2` |
| `postprocessing.py:585` | `(yy - 150) * 2 + (xx - 150) * 2 <= 20 ** 2` | `(yy - 150) ** 2 + (xx - 150) ** 2 <= ...` |

The last two were silently wrong maths, not import errors — the gyre envelope
was computing `-2·r / (4·L)` instead of a Gaussian.

**Check your own repo against this list.** If your local copies are fine, the
mangling happened during upload and you only need the two `**` fixes plus the
`_debug` key fix. No physics or model logic was otherwise touched.

Verification after the fixes:

```
> python run_module2.py --self-test
[self-test] most_likely_origin error vs. true origin: 0.18 km
[self-test] true hour ranks #1/12
[self-test] OK
```

---

## 2. What was added

Nothing in `advection.py`, `backtrack.py`, `config.py`, `unet_model.py`,
`preprocessing.py` or `postprocessing.py` was modified beyond those repairs.
The new work sits alongside them.

| New file | Why |
|---|---|
| `module1_detection/png_inference.py` | A second entry into the trained U-Net for the curated PNG catalog. `run_module1.py` needs a calibrated 2-band GeoTIFF plus a metadata sidecar, a land shapefile and `weights/lookalike_xgb.json`; the demo tiles are single-channel PNGs with none of that. Same weights, same `/255` normalisation, simpler pre/post-processing. |
| `module2_backtracking/forward_track.py` | Section 6b's missing forward mode. Drives the existing integrator with `direction="forward"` from `most_likely_origin`. No new physics. |
| `OpenMeteoGriddedCurrentField` in `shared_apis/drift_forcing_client.py` | See section 3 — this one matters. |
| `backend/*` | The three services and the gateway. |
| `frontend/*` | The dashboard. |
| `scripts/*` | Catalog builder, database seeder, health check, Windows launchers. |

---

## 3. Two integration decisions that change the results

These are the two places where wiring the models together exposed a problem
that would have made the demo look like it worked while producing meaningless
numbers. Both are worth knowing about before you present this.

### 3a. A spatially uniform current field flattens the posterior

`OpenMeteoCurrentField` queries one point and returns that velocity everywhere.
Its own docstring says so. In a spatially uniform flow:

- a particle cloud advected backward `h` hours then forward `h` hours returns
  exactly where it started, for every `h`, so `centroid_offset_km` is ~0 for
  every candidate;
- with no velocity gradient the cloud's area and elongation are conserved
  exactly, so those two ABC terms are identical for every candidate too.

All three terms of `backtrack.py`'s likelihood then coincide, the posterior
comes out flat, and `most_likely_origin` degrades into an unweighted average
along the drift track. Nothing crashes. It just stops being Bayesian.

`OpenMeteoGriddedCurrentField` samples a 4×4 grid across the search bbox in
**one** batched request — Open-Meteo accepts comma-separated coordinate lists —
and bilinearly interpolates. Same data, same API call count, but with a real
velocity gradient for the ABC kernel to work with. It is tried first;
single-point and then synthetic remain as fallbacks. Set `OPENMETEO_GRID_SIDE=1`
in `.env` to go back to the old behaviour.

### 3b. The release footprint has to be sized from the observation

`backtrack.py` scores candidates partly on `log(sim_area / obs_area)`.
`sim_area` comes from advecting a circle of `initial_release_radius_km` forward,
and advection is near area-preserving, so it stays close to that circle's area.

Module 2's default is 0.3 km. Real detections here come out around 30 km²,
whose equivalent radius is 3.2 km. That gives an area ratio near 0.01, so
`log(ratio) ≈ -4.6`, divided by `sigma_log_area` 0.6 is -7.7, squared is 59, and
`exp(-29.5)` underflows to zero. **Every** candidate scores zero,
`degenerate_ensemble` fires, and the posterior falls back to uniform.

The backtrack service therefore sizes the seed circle from the detection:
`radius = sqrt(area_km2 / π) × RELEASE_FOOTPRINT_FRAC`. Measured difference on
`demo_img_01`:

| Seed radius | Effective sample size | Result |
|---|---|---|
| 0.30 km (Module 2 default) | 1.9 / 24 | all mass on one candidate, origin 35 km off |
| 3.19 km (fitted to the observation) | 12.9 / 24 | well-spread posterior, ±3.7 km |

Tune with `RELEASE_FOOTPRINT_FRAC` in `.env`. At 1.0 the area term is neutral
and discrimination comes from centroid offset and elongation, which is the
honest position for an integrator that models no spreading or weathering.

---

## 4. Setup on Windows

### 4.1 Prerequisites

| | Version | Check |
|---|---|---|
| Python | 3.10–3.12 | `python --version` |
| Node.js | 18+ | `node --version` |
| MongoDB Community Server | 6.0+ | `mongod --version` |

Python must be on PATH. If `python` opens the Microsoft Store, uncheck both
Python entries under Settings → Apps → Advanced app settings → App execution
aliases.

### 4.2 Install MongoDB locally

There is no "create the database" step. MongoDB creates databases and
collections lazily on first write, so starting the server and running the seed
script is the whole thing.

1. Download the MSI: <https://www.mongodb.com/try/download/community>
   (Platform: Windows, Package: msi).
2. Run it, choose **Complete**, and leave **Install MongoDB as a Service**
   ticked with "Run service as Network Service user". Compass is optional but
   handy for looking at the data.
3. Confirm the service is up:

   ```
   sc query MongoDB
   ```

   Look for `STATE : 4 RUNNING`. If it is stopped:

   ```
   net start MongoDB
   ```

   Both commands need an Administrator prompt.

If you would rather not install it as a service, run it in the foreground and
leave the window open:

```
mkdir C:\data\db
"C:\Program Files\MongoDB\Server\8.0\bin\mongod.exe" --dbpath C:\data\db
```

Adjust `8.0` to whatever version you installed.

To confirm you can talk to it, open a second terminal:

```
"C:\Program Files\MongoDB\Server\8.0\bin\mongosh.exe"
> db.runCommand({ ping: 1 })
> exit
```

`mongosh` is a separate download if the MSI did not include it, but nothing in
this project needs it — it is only for poking around.

### 4.3 Python environment

From the project root:

```
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` may pull a CUDA build of PyTorch, which is a large download
and buys nothing here — the U-Net does one 256×256 forward pass. For the small
CPU wheel instead:

```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 4.4 Node dependencies

```
cd backend\gateway
npm install
cd ..\..\frontend
npm install
cd ..
```

### 4.5 Configuration

Copy the template. Every value in it is already the default, so an unedited
copy works:

```
copy .env.example .env
```

**On the Open-Meteo key.** The URL you sent is the public endpoint, not a key —
`https://marine-api.open-meteo.com/v1/marine`. It is already hardcoded in
`shared_apis/drift_forcing_client.py` and the free tier needs no account and no
key (10,000 calls/day, plenty for a demo). Leave `OPENMETEO_API_KEY` blank.

Only fill it in if you buy a commercial plan — a key switches requests to
Open-Meteo's separate paid host, and sending a free-tier request there fails.

### 4.6 Build the catalog and seed the database

```
python scripts\make_demo_catalog.py
python scripts\seed_db.py
```

The first runs the U-Net on each demo tile to find where its slick is, then
fits each scene's geo bbox so that slick lands on a chosen coordinate near the
vessel dataset. Details in section 6.

The second creates the `oilspill` database with three collections and loads
`demo_images` and `ships`. Re-running it is safe; it replaces those two
collections and leaves previous pipeline runs alone. Add `--reset-events` to
clear those too.

### 4.7 Preflight

```
python scripts\check_health.py
```

This is the one command to run before a demo. It checks packages, MongoDB and
its collections, the U-Net checkpoint, and — most importantly — makes a live
call to Open-Meteo and tells you whether you would get real current data or the
synthetic fallback. It exits non-zero if anything needs fixing.

### 4.8 Run

```
scripts\start_all.bat
```

Five windows open. Give the detection service about fifteen seconds to load the
checkpoint, then open <http://localhost:5173>.

To start things by hand instead, one terminal each from the project root:

```
python backend\detection_service\main.py     :8001
python backend\backtrack_service\main.py     :8002
python backend\ais_service\main.py           :8003
cd backend\gateway && node server.js         :4000
cd frontend && npm run dev                   :5173
```

`scripts\stop_all.bat` frees all five ports if a window was closed without the
process exiting.

---

## 5. Using it

1. Pick a scene from the four thumbnails.
2. Press **Run analysis**.
3. The map fills in a stage at a time over Socket.io:
   - the detected slick outline in amber;
   - the candidate-origin cloud sweeping oldest to newest, settling into the
     full posterior with an uncertainty ring;
   - each vessel pulsing as it is checked, then the ranked panel.
4. Pick a duration and press **Forecast** to advect the estimated release point
   forward.

Click any posterior cell or vessel marker for its numbers. Clicking a vessel row
expands its scoring breakdown.

A full run takes roughly 5–8 seconds: about 2.5 s for detection (first run
includes model load), 1–3 s for the drift ensemble, well under a second for
correlation.

### Reading the confidence badge

It reports the drift model's own diagnostics, and each failure gets its own
message because "low confidence" without a reason is not useful:

- **Not physically meaningful** — the current API was unreachable and this ran
  on `SyntheticGyreField`. The pipeline is being demonstrated; the origin is
  not a result. Fix your network and re-run.
- **Ensemble collapsed** — every candidate scored near zero and the posterior
  went uniform. The 1–48 h search window probably does not bracket the release.
- **Origin uncertain** — effective sample size below 15% of the ensemble, so
  nearly all mass sits on one hypothesis. Read `uncertainty_radius_km` with
  caution.
- **Weak current gradient** — the field is nearly uniform across the search
  area, so release times are hard to tell apart. Space is better constrained
  than time.

---

## 6. About the demo data

### The SAR tiles

`scene_01_delta.png` is the VV panel from the tile you supplied, cropped out of
the matplotlib figure and resized to 512×512. The U-Net reproduces its ground
truth mask closely — the two largest components trace the same branching slick
structure, at 0.88 mean confidence inside the mask.

The other three are geometric variants of it (rotate 90°, mirror, rotate 270 and
crop). Same real SAR texture re-oriented, so the network is still segmenting a
genuine slick signature rather than something invented. Each produces a
different mask geometry and therefore a different detection.

To use real Kaggle tiles, drop the PNGs into `data/demo_images/`, edit the
`SCENES` list at the top of `scripts/make_demo_catalog.py`, and re-run it
followed by `seed_db.py`.

### Why the bboxes are fitted rather than fixed

The Kaggle tiles carry no CRS, no geotransform and no acquisition time. The
build spec's answer is to register each scene with a bbox and a timestamp by
hand and interpolate pixel coordinates into it.

Doing that by hand is fiddly: pick a bbox, run detection, find the slick landed
40 km from your vessels, adjust. `make_demo_catalog.py` inverts it — you state
where the slick should be, it runs the U-Net, finds the primary component's
centroid, and solves for the bbox that puts it there. The bbox is demo metadata
either way; this just makes the assignment reproducible and guarantees the
detection, the drift posterior and the AIS dataset share one patch of ocean.

Each scene spans 0.153° × 0.135°, about 15 × 15 km at 28.4 °N, which puts the
detected slick at a plausible 29–34 km² rather than the hundreds a wider tile
would imply.

### Geographic coherence

Everything sits in the Mississippi Canyon area of the Gulf of Mexico, matching
`ships_seed.json` (7 vessels, 28.17–29.31 °N, 89.43–88.99 °W, tracks running
2026-09-03 16:12 Z to 2026-09-04 06:33 Z):

| Scene | Slick centre | Imaged |
|---|---|---|
| Mississippi Canyon | 28.310 N, 89.020 W | 06:12 Z |
| Mars Ridge | 28.455 N, 89.145 W | 05:30 Z |
| Shelf Edge | 28.235 N, 89.190 W | 04:45 Z |
| West Flank | 28.575 N, 89.255 W | 06:40 Z |

Each is inside the vessel field, so a 1–48 h backtrack lands on water where
several vessels were present and the scoring has something to discriminate.

---

## 7. How the vessel scoring works

For every reported position of a vessel inside `release_time_window`:

```
grid_proximity = Σ over cells c of origin_probability_grid:
                   p(c) · exp( −distance_km(vessel, c) / SIGMA_DIST_KM )
```

`SIGMA_DIST_KM` is imported from Module 2's `config.py` (default 8.0), so
"close" means the same thing here as inside the ABC kernel that produced the
posterior. Change it in one place and both move together.

Scoring against the whole grid rather than the point estimate is the point: a
vessel on a broad ridge of moderate probability is a better lead than one near
the mean of a bimodal posterior, which may be water no candidate favoured.

The raw sum is bounded by 1.0 and in practice much lower, so it is normalised
against what an imaginary vessel sitting exactly on `most_likely_origin` would
score. That makes the number readable as "how close to ideal is this position",
and unlike normalising by the best actual vessel it does not hand out a 100 to
the least-bad candidate in a fleet that was all 200 km away.

```
time_centrality  = 1 − 2·|t − window_centre| / window_width,  clamped to [0,1]
suspicion_score  = 100 · (0.7 · normalised_proximity + 0.3 · time_centrality)
```

Vessels with no position inside the window are still returned, with
`time_centrality` 0 and `in_release_window` false, scored on their nearest
in-time position. "Checked, and here is why it is not them" is a result, and the
map animation shows every vessel being considered.

A representative run:

```
 57.2  SANCO SWORD        Other        3.7 km   outside window
 49.6  CARNIVAL VALOR     Passenger    5.9 km
 25.1  MARS TLP           Other       27.8 km
 21.6  OVERSEAS MARTINEZ  Tanker      23.3 km
 14.2  LEGACY             Towing      28.0 km
  4.5  OSAKANA            Cargo       24.6 km   outside window
  2.4  YAS                Tanker      38.5 km
```

A high score means a vessel was in the right water at the right time according
to a drift model with real uncertainty. It is an investigative lead. It is not
evidence of discharge, and the API returns no field that phrases it as one.

---

## 8. API reference

Everything goes through the gateway on port 4000. Vite proxies `/api` and the
socket handshake, so the frontend only ever talks to its own origin.

| Method | Route | Purpose |
|---|---|---|
| GET | `/api/scenes` | Catalog for the picker |
| GET | `/api/scenes/:id/image` | Tile PNG |
| POST | `/api/pipeline/run` | `{image_id}` → the whole chain |
| POST | `/api/drift/forward` | Forecast from an origin |
| GET | `/api/events/:id` | A previous run from MongoDB |
| GET | `/api/health` | Fan-out check across all services |

Socket.io events, each carrying its payload rather than just a name — so a
dropped socket degrades into "the map fills in at the end" instead of an empty
screen:

```
stage:started          {stage}
stage:detection_done   {detection}
stage:backtrack_done   {backtrack}
stage:ais_done         {ranked_vessels, ais_debug}
stage:forward_done     {forward}
stage:pipeline_done    {elapsed_seconds}
stage:error            {stage, message}
```

The Python services are reachable directly on 8001–8003 if you want to test one
in isolation; each serves interactive docs at `/docs`.

### Collections

| Collection | Contents |
|---|---|
| `demo_images` | Curated catalog: bbox, timestamp, provenance |
| `spill_events` | One document per run — detection, posterior, vessels, forecast |
| `ships` | Vessel tracks, plus a `2dsphere` index on `track_geometry` |

The AIS scorer walks paths in Python rather than querying by geometry; at seven
vessels that is faster than a round trip. The index is there so a larger dataset
can switch to a `$near` pre-filter without a schema change.

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

Change them in `.env`; `vite.config.js` also hardcodes the gateway's port in its
proxy target if you move that one.

---

## 10. Troubleshooting

**"Cannot reach the gateway"** — the gateway is not running, or Mongo was down
when it started. Start `mongod`, then restart the gateway.

**"The ships collection is empty"** — `python scripts\seed_db.py`.

**Confidence badge says "Not physically meaningful"** — Open-Meteo was
unreachable. Check your connection and run `python scripts\check_health.py`,
which tests the API directly. Corporate networks and VPNs sometimes block it.

**Map is blank but the panels have data** — the CARTO basemap needs internet.
The app detects the failure and falls back to a flat ocean fill; deck.gl layers
still render. Everything except the coastline still works offline.

**Detection returns 422 "no slick above threshold"** — lower it in `.env` with
`DETECT_THRESHOLD=0.35`, or pick another scene.

**Port already in use** — `scripts\stop_all.bat`.

**Torch install is enormous** — use the CPU index URL in section 4.3.

**Detection service takes ~15 s to start** — that is the 31 MB checkpoint
loading. It is warmed at startup so the first request does not pay for it.

---

## 11. A note on the bundle

`module1_detection/weights/unet_last.pt` (93 MB) is **not** included, to keep
the download reasonable. Only `unet_best.pt` (31 MB, epoch 12, val IoU 0.651)
is loaded by anything here. Copy `unet_last.pt` back in from your own repo if
you want it for reference.

`node_modules/` is not included either — run `npm install` in both
`backend/gateway` and `frontend` as in section 4.4. The lockfiles are included,
so you get the exact versions this was tested against.

---

## 12. Known limits

- **Wind is ignored.** `wind_u_mps`/`wind_v_mps` default to 0 and no wind API is
  called. This is Module 2's intended posture, not an oversight — but real
  leeway on a surface slick is 2–4% of wind speed and is not modelled here.
- **The current grid is coarse.** Four points across a ~1.6° box resolves
  large-scale shear, not mesoscale eddies. A CMEMS NetCDF via `NetCDFGridField`
  is the higher-fidelity path; point `CURRENT_CACHE_PATH` at one if you have it.
- **No diffusion, spreading or weathering.** Advection is deterministic, so a
  patch's area is nearly conserved. This is why the release footprint has to be
  sized from the observation (section 3b), and why forecast spread is smaller
  than real forecast uncertainty.
- **The demo tiles are single-polarisation.** VV is duplicated into the VH
  channel because the PNGs have one band. The network was trained on genuine
  dual-pol input, so it has less to work with than an operational scene.
- **AIS is synthetic.** `ships_seed.json` mixes real vessel identities with
  invented tracks. Two entries are flagged `synthetic: true`.
- **The bboxes are assigned, not measured.** See section 6.
