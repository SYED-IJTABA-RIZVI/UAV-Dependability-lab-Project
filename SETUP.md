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

Then bring up everything — app, Postgres, YOLO, and the shared VLM service —
with:

```bash
docker compose --profile gpu up -d --build
```

Only YOLO loads its checkpoint(s) at startup. The `vlm` service loads nothing
until the first `/classify` call, and only ever keeps ONE of
InternVL2.5/DeepSeek-VL/BLIP-2 in VRAM at a time — switching which VLM you
select in the app unloads whatever was loaded and loads the new one, which
takes real time (an 8B checkpoint took ~25s+ in testing) on that first call
after switching. This is deliberate: three 8-bit VLMs loaded simultaneously
alongside YOLO doesn't reliably fit on a 20GB card; trading first-call
latency for guaranteed VRAM headroom does. Check each service is actually up:

```bash
curl localhost:8501                    # app
docker compose exec yolo curl -s localhost:8000/health
docker compose exec vlm curl -s localhost:8000/health   # loaded_model_id is null until first use
```

### Known unknowns on first real run

This was built and syntax-checked on a machine with no GPU — the pieces below
were written against each library's documented API but **not run
end-to-end**, so budget time to debug on first launch:

- **VRAM fit**: YOLO (both checkpoints, always resident) + whichever single
  VLM is currently loaded need to fit on one RTX 4000/20GB card. Current
  checkpoints: `InternVL2_5-8B`, `deepseek-vl-7b-chat`, `blip2-opt-2.7b` — the
  lazy-load/unload scheme in `app/vlm_server/server.py` exists specifically
  because loading all three simultaneously did not fit. If even one VLM
  loaded at a time doesn't fit alongside YOLO, swap in a smaller checkpoint
  via `real_inference.py`'s `VLM_MODEL_IDS` mapping (the actual HF model id
  used per VLM name).
- **InternVL2.5 loading code** (`app/vlm_server/server.py`,
  `_load_generic_trust_remote_code` / `_infer_generic_chat`): loaded via
  `trust_remote_code=True` per its HF model card. If loading or `.chat()`
  fails, that function is the fix point — check the model's actual HF page
  for its current usage example. Confirmed on the lab GPU host: needs
  `protobuf` installed (now in requirements.txt) for its sentencepiece-based
  tokenizer. If you still hit `RuntimeError: piece must not include null
  character` after that, the cached tokenizer file itself is likely
  corrupted/incomplete (e.g. from a disk-full interruption mid-download) —
  clear that model's entry from the `hf_cache` volume and let it re-download.
- **DeepSeek-VL (v1) loading code** (`_load_deepseek_vl` / `_infer_deepseek_vl`
  in `app/vlm_server/server.py`): confirmed on the lab GPU host that the
  generic `trust_remote_code` path does NOT work for it (raises
  `KeyError: 'multi_modality'`) — it needs its own `deepseek_vl` pip package
  (installed from their GitHub repo in requirements.txt) and a dedicated
  VLChatProcessor/MultiModalityCausalLM loading path, which is now what this
  file does. That specific fix is written directly from DeepSeek-VL's
  published usage example but hasn't itself been run on the GPU host yet —
  the `deepseek_vl` package's own dependencies could also conflict with the
  `transformers<4.52` pin above; if the pip install step itself fails, that's
  the first thing to check.
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
