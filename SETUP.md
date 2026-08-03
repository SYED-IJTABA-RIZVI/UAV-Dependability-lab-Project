# Setup

## Everyday use (any machine, no GPU needed)

```bash
cp .env.example .env      # first time only
docker compose up -d --build
```

Open http://localhost:8501. This runs the Streamlit app + Postgres. Any
inference that isn't configured (no RFDETR checkpoint, no VLM key/service)
automatically falls back to the built-in mock simulator — the app always
works, even with nothing else set up.

To also get real **Qwen2.5-VL** results (the one VLM with a hosted API),
add your OpenRouter key to `.env`:

```
OPENROUTER_API_KEY=sk-or-...
```

Get one at https://openrouter.ai/keys. No rebuild needed, just restart:
`docker compose up -d`.

## Full local stack (RTX 4000 / GPU machine)

InternVL3, DeepSeek-VL, and BLIP-2 have no hosted API anywhere — they run
locally as their own containers, plus the real RFDETR model. One-time setup
on the GPU machine:

1. **Install `nvidia-container-toolkit`** on the host (lets Docker containers
   see the GPU) — see NVIDIA's install docs for your OS. Confirm it works:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```
2. **Drop the two RFDETR checkpoints** into `weights/`:
   ```
   weights/rfdetr_rgb.pt
   weights/rfdetr_ir.pt
   ```
   (filenames must match exactly — see `RFDETR_RGB_CHECKPOINT_PATH` /
   `RFDETR_IR_CHECKPOINT_PATH` in `docker-compose.yml` if you want different names)
3. **Set `OPENROUTER_API_KEY`** in `.env` (same as above, for Qwen2.5-VL).

Then bring up everything — app, Postgres, RFDETR, and all 3 local VLMs — with:

```bash
docker compose --profile gpu up -d --build
```

First run will be slow: each VLM container downloads its model weights from
Hugging Face (several GB each) into a shared cache volume, so subsequent
restarts are fast. Check each service is actually up:

```bash
curl localhost:8501                    # app
docker compose exec rfdetr curl -s localhost:8000/health
docker compose exec vlm-internvl3 curl -s localhost:8000/health
docker compose exec vlm-deepseek-vl curl -s localhost:8000/health
docker compose exec vlm-blip2 curl -s localhost:8000/health
```

### Known unknowns on first real run

This was built and syntax-checked on a machine with no GPU — the pieces below
were written against each library's documented API but **not run
end-to-end**, so budget time to debug on first launch:

- **VRAM fit**: RFDETR + 3 VLMs loaded simultaneously (8-bit) need to fit on
  one RTX 4000. Smallest available checkpoint was picked for each VLM
  (`InternVL3-8B`, `deepseek-vl2-tiny`, `blip2-opt-2.7b`) to maximize the odds,
  but if it doesn't fit: run only the VLM(s) you're actively using (comment
  out the other `vlm-*` services, or `docker compose --profile gpu up -d
  --build app postgres rfdetr vlm-internvl3` to start a subset), or swap in
  smaller checkpoints via each service's `MODEL_ID` in `docker-compose.yml`.
- **InternVL3 / DeepSeek-VL loading code** (`app/vlm_server/server.py`,
  `_load_generic_trust_remote_code` / `_infer_generic_chat`): loaded via
  `trust_remote_code=True` per their HF model cards. If loading or `.chat()`
  fails, that function is the fix point — check the model's actual HF page
  for its current usage example.
- **RF-DETR class-id order** (`app/rfdetr_server/server.py`): assumes the
  checkpoint's class ids are `0=Drone, 1=Bird, 2=Helicopter, 3=Airplane`,
  matching this app's convention everywhere else. If detections come back
  mislabeled, fix the `CLASSES` list order in that file to match how the
  model was actually trained.

If a service fails to start or a model won't load, the app still works fine
via mock fallback for everything else — nothing else breaks.
