# SkyEye

**A real-time sky object detection and classification system for UAV dependability research.**

SkyEye watches an image (or a batch of images) and answers one question: *what is that object in the sky?* It classifies aerial objects into one of four categories — **Airplane**, **Bird**, **Drone**, or **Helicopter** — using a two-stage pipeline that combines a fast object detector with a vision-language model as an intelligent fallback.

---

## The problem

Distinguishing a drone from a bird, or a small UAV from a distant airplane, is a genuinely hard visual classification problem. It matters for a very practical reason: counter-UAS (counter-drone) systems, airspace monitoring, and UAV dependability research all depend on being able to reliably tell "this is an actual drone" apart from "this is a bird that looks like one from this angle and distance." Get it wrong in one direction and a real drone goes unflagged; get it wrong in the other and every bird triggers a false alarm.

A single fixed-function object detector (like YOLO) is fast but brittle — it's excellent when confident, and it doesn't know when to doubt itself. A large vision-language model is much better at genuinely *reasoning* about an ambiguous or unusual image, but it's too slow to run on every single frame.

SkyEye's answer is a **cascade**: let the fast detector handle the easy, high-confidence cases instantly, and only escalate to the slower, smarter model when the fast detector isn't sure. This is the same tradeoff real-time detection systems make in practice — optimize for speed on the common case, spend the extra compute only where it's actually needed.

## What SkyEye does

- **Detects and localizes** objects in RGB or IR/thermal imagery using a YOLO checkpoint (bounding box + class + confidence), with every detected object boxed — not just the most confident one.
- **Classifies** the same image using a vision-language model (VLM) that reasons over the full visual context and explains its answer in plain language.
- **Cascades between the two** automatically: run YOLO first (fast), and only call the VLM when YOLO's confidence falls below a configurable threshold — or run either one alone, or run the VLM on every image for maximum accuracy at the cost of latency.
- **Never fabricates a result.** If a model isn't configured or a real inference call fails, the app shows a clear "model unavailable" message instead of silently guessing — a wrong-looking result is worse than no result.
- **Persists every run** (single image or batch folder) to Postgres, so past results, detections, and annotated images stay queryable through a History view.

## Detection classes

`Airplane` · `Bird` · `Drone` · `Helicopter`

---

## Architecture

SkyEye runs as a small set of Docker Compose services:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────────┐
│             │─────▶│    yolo      │      │                 │
│             │      │  (YOLOv12    │      │   PostgreSQL    │
│    app      │      │  RGB /       │      │  (run history,  │
│ (Streamlit) │      │  YOLOv10 IR) │      │   detections)   │
│             │      └──────────────┘      │                 │
│             │─────▶┌──────────────┐      └─────────────────┘
│             │      │     vlm      │              ▲
│             │      │ (InternVL2.5*/│              │
│             │      │  DeepSeek-VL/│──────────────┘
│             │      │  BLIP-2)     │
│             │      └──────────────┘
│             │
│             │─────▶ OpenRouter API (Qwen2.5-VL, hosted)
└─────────────┘
```
<sub>* InternVL2.5 is implemented but currently disabled in the UI — see [Known limitations](#known-limitations).</sub>

**`app`** — the Streamlit UI and orchestration layer. Stays GPU-reservation-free so `docker compose up` works on any machine, with or without a GPU; if the detection services aren't reachable, the app tells you so directly rather than pretending to work.

**`yolo`** — a FastAPI service wrapping [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO. Loads two checkpoints at startup — one for RGB, one for IR/thermal — and keeps both resident in VRAM (they're comparatively small). Runs class-agnostic non-max suppression on top of Ultralytics' own per-class NMS, since a single real object can otherwise get boxed multiple times under different predicted classes.

**`vlm`** — a single FastAPI service shared by all self-hosted VLMs (currently DeepSeek-VL and BLIP-2; InternVL2.5 is implemented but disabled — see below). Rather than keeping every VLM loaded in VRAM simultaneously (which doesn't fit on a single consumer GPU), it lazy-loads whichever model was actually requested and unloads the previous one on a switch. All three models' weights are pre-fetched to local disk at container startup so a later switch only pays the (comparatively fast) VRAM-load cost, not a fresh multi-gigabyte download.

**Qwen2.5-VL** is the exception — it's called directly from `app` via [OpenRouter](https://openrouter.ai)'s hosted API, since it's the one VLM here with no local GPU footprint at all.

**`postgres`** — stores every run (single image or folder batch): which files were processed, every detection (class, confidence, bounding box, latency, which model produced it), and whether/why a result got escalated from YOLO to a VLM.

### Real inference only — no silent fallback

Every model call either succeeds with a real result or the app tells you clearly that it didn't. There's a mock simulator in the codebase for exercising the rest of the app's logic on a machine with no GPU, but it's opt-in only (`ALLOW_MOCK_FALLBACK=1`) and is never enabled in the shipped `docker-compose.yml` — the deployed tool always requires real models.

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| Object detection | [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (YOLOv12 for RGB, YOLOv10 for IR) |
| Vision-language models | InternVL2.5-8B, DeepSeek-VL-7B-Chat, BLIP-2 (self-hosted), Qwen2.5-VL (via OpenRouter) |
| Model serving | FastAPI + Uvicorn |
| Database | PostgreSQL 16 + SQLAlchemy |
| Deployment | Docker Compose |
| GPU inference | PyTorch + CUDA 12.1, `bitsandbytes` 8-bit quantization |

---

## Project structure

```
.
├── docker-compose.yml       # postgres, app, yolo, vlm services
├── .env.example             # config template — copy to .env
├── SETUP.md                 # detailed GPU-host setup + known-unknowns log
├── weights/                 # your YOLO checkpoints go here (gitignored)
├── data/                    # default mount point for FOLDER batch inputs
└── app/
    ├── main.py               # Streamlit UI — IMAGE / FOLDER / HISTORY tabs
    ├── mock_backend.py        # inference dispatch (real-first, mock opt-in only)
    ├── real_inference.py      # calls out to yolo/vlm services + OpenRouter
    ├── batch.py               # FOLDER batch orchestration (CSV + annotated images)
    ├── db.py                  # Postgres models + persistence
    ├── drawing.py              # bounding-box rendering
    ├── labels.py               # shared file-discovery helpers
    ├── style.py                # UI theme (dark, black/blue/grey)
    ├── yolo_server/            # standalone YOLO detection service
    └── vlm_server/              # standalone shared VLM service (lazy-load/unload)
```

---

## Getting started

### Quick start (no GPU needed)

```bash
cp .env.example .env
docker compose up -d --build
```

Open **http://localhost:8501**. This runs the app + Postgres. Without a GPU host configured, YOLO/local-VLM calls will show a clear "model unavailable" message — the UI itself, the database, and the History view all work regardless.

To also get real results from **Qwen2.5-VL** (the one VLM with a hosted API), add an [OpenRouter](https://openrouter.ai/keys) key to `.env`:
```
OPENROUTER_API_KEY=sk-or-...
```
then `docker compose up -d` again — no rebuild needed.

### Full setup (GPU machine — real YOLO + local VLMs)

See **[SETUP.md](SETUP.md)** for the complete walkthrough: installing `nvidia-container-toolkit`, placing YOLO checkpoints in `weights/`, and bringing up the full stack with:

```bash
docker compose --profile gpu up -d --build
```

---

## Usage

**IMAGE** — upload a single image, pick a pipeline mode and confidence cutoff, hit **Analyze Frame**. Every object YOLO detects gets its own bounding box; the final classification (YOLO's or the VLM's, depending on the mode and whether escalation triggered) shows in the workspace footer.

**FOLDER** — point at a folder of images (must be visible inside the container — see the path-mounting note in `.env.example`), get a CSV of per-image results plus every annotated image written to your chosen output folder.

**HISTORY** — browse and drill into any past IMAGE or FOLDER run, including every individual detection recorded for it.

### Pipeline modes

- **YOLO Only** (`EDGE-FAST`) — detection only, lowest latency, no fallback.
- **VLM Only** (`DEEP-VLM`) — every image goes to the VLM, highest contextual accuracy, slowest.
- **YOLO & VLM (Adaptive Fallback)** — YOLO classifies instantly; anything below the confidence cutoff gets re-checked by the VLM, which becomes the source of truth for that result.

---

## Configuration

All configuration lives in `.env` (copy from `.env.example`). Key variables:

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | Enables real Qwen2.5-VL results |
| `DATASETS_PATH` | Host path mounted into the app container at `/datasets`, for FOLDER inputs living outside the project directory |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | Database credentials |

GPU-host-only settings (checkpoint paths, VLM model IDs) live directly in `docker-compose.yml` / `app/vlm_server/server.py` — see `SETUP.md`.

---

## Known limitations

- **InternVL2.5 is implemented but not currently offered in the UI.** It has an unresolved loading bug (its tokenizer fails to initialize correctly under `trust_remote_code`) that hasn't been root-caused yet. The code stays in `vlm_server/server.py`, ready to re-enable once fixed.
- **First use of a local VLM after switching models is slow** (a full VRAM load, not just a disk read) — by design, since only one local VLM stays resident in VRAM at a time. See the Architecture section above.
- **No ground-truth evaluation mode.** Accuracy/precision-recall metrics against labeled datasets aren't currently part of the app — it's detection + classification only.
