# Setup

## Everyday use (any machine, no GPU needed)

```bash
cp .env.example .env      # first time only
docker compose up -d --build
```

Open http://localhost:8501. This runs the Streamlit app + Postgres. Real
inference is required by default — if a model isn't configured (no YOLO
checkpoint, no VLM key/service), the app shows a clear "MODEL UNAVAILABLE"
error rather than a result, instead of silently substituting a fake one. The
built-in mock simulator still exists for local development on a machine with
no GPU, but only runs if explicitly enabled with `ALLOW_MOCK_FALLBACK=1` —
never set in `docker-compose.yml`, so the deployed app always requires real
models.

To also get real **Qwen2.5-VL** results (the one VLM with a hosted API),
add your OpenRouter key to `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

Get one at https://openrouter.ai/keys. No rebuild needed, just restart:
`docker compose up -d`.

## Full local stack (RTX 4000 / GPU machine)

InternVL2.5, DeepSeek-VL, and BLIP-2 have no confirmed free hosted API
anywhere — they run locally as their own containers, plus the real YOLO
models (YOLOv12 for RGB, YOLOv10 for IR/Thermal). One-time setup on the GPU
machine:

1. **Install `nvidia-container-toolkit`** on the host (lets Docker containers
   see the GPU) — see NVIDIA's install docs for your OS. Confirm it works:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
2. **Drop the two YOLO checkpoints** into `weights/`:
   ```
   weights/yolov12_rgb.pt
   weights/yolov10_ir.pt
   ```
   (filenames must match exactly — see `YOLO_RGB_CHECKPOINT_PATH` /
   `YOLO_IR_CHECKPOINT_PATH` in `docker-compose.yml` if you want different names.
   `.pt`/`.pth` are interchangeable — Ultralytics doesn't care about the extension.)
3. **Set `OPENROUTER_API_KEY`** in `.env` (same as above, for Qwen2.5-VL).

Then bring up everything — app, Postgres, YOLO, and all 3 local VLMs — with:

```bash
docker compose --profile gpu up -d --build
```

First run will be slow: each VLM container downloads its model weights from
Hugging Face (several GB each) into a shared cache volume, so subsequent
restarts are fast. Check each service is actually up:

```bash
curl localhost:8501                    # app
docker compose exec yolo curl -s localhost:8000/health
docker compose exec vlm-internvl3 curl -s localhost:8000/health
docker compose exec vlm-deepseek-vl curl -s localhost:8000/health
docker compose exec vlm-blip2 curl -s localhost:8000/health
```

### Known unknowns on first real run

This was built and syntax-checked on a machine with no GPU — the pieces below
were written against each library's documented API but **not run
end-to-end**, so budget time to debug on first launch:

- **VRAM fit**: YOLO (both checkpoints) + 3 VLMs loaded simultaneously (8-bit)
  need to fit on one RTX 4000. Current checkpoints: `InternVL2_5-8B`,
  `deepseek-vl-7b-chat`, `blip2-opt-2.7b`. If it doesn't fit: run only the
  VLM(s) you're actively using (comment out the other `vlm-*` services, or
  `docker compose --profile gpu up -d --build app postgres yolo
  vlm-internvl3` to start a subset), or swap in smaller checkpoints via each
  service's `MODEL_ID` in `docker-compose.yml`.
- **InternVL2.5 / DeepSeek-VL loading code** (`app/vlm_server/server.py`,
  `_load_generic_trust_remote_code` / `_infer_generic_chat`): loaded via
  `trust_remote_code=True` per their HF model cards. If loading or `.chat()`
  fails, that function is the fix point — check the model's actual HF page
  for its current usage example. DeepSeek-VL (v1) in particular has
  historically shipped its own `deepseek_vl` pip package with a dedicated
  `VLChatProcessor` rather than working through plain
  `AutoModel`/`AutoTokenizer` — check that first if it specifically fails.
- **YOLOv12 support in `ultralytics`**: YOLOv12 is newer than YOLOv10/v11;
  `app/yolo_server/requirements.txt` pins `ultralytics>=8.3`, but confirm the
  installed version actually supports loading a YOLOv12 checkpoint — if
  `YOLO(path)` fails to load the RGB checkpoint, check the `ultralytics`
  version against their YOLOv12 release notes and bump the pin if needed.
- **Class names**: `app/yolo_server/server.py` reads class names directly from
  each checkpoint's own embedded metadata (`result.names`), not a hardcoded
  list — so unlike the earlier RFDETR setup, this shouldn't need a manual fix.
  If detections still come back mislabeled, the checkpoint's embedded names
  don't match this app's expected set (`Airplane`, `Bird`, `Drone`,
  `Helicopter`) — check what `result.names` actually contains for that
  checkpoint.

If a service fails to start or a model won't load, the app shows a clear
"MODEL UNAVAILABLE" error for that specific model instead of a fabricated
result — everything else (other models, other tabs) still works normally.
