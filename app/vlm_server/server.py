"""
Self-hosted VLM classification server, shared by InternVL2.5-8B,
DeepSeek-VL-7B-Chat, and BLIP-2. Deliberate VRAM tradeoff: only ONE of these
three is ever loaded at a time (plus YOLO, which stays resident in its own
container) — each /classify request names which model_id it wants, and this
service lazy-loads it, unloading whatever was previously loaded first if it's
a different model. Switching models costs real time (an 8B checkpoint took
~25s+ to load in testing) but keeps VRAM usage to one VLM's footprint instead
of three simultaneously, which is what a 20GB card doesn't have room for.

BLIP-2's `transformers` API is stable and should just work. InternVL2.5 is
loaded via generic `AutoModel`/`trust_remote_code=True`, per its HF model
card. DeepSeek-VL (v1, deepseek-ai/deepseek-vl-7b-chat) does NOT work through
that generic path — confirmed on the lab GPU host: it raised
`KeyError: 'multi_modality'` / "Transformers does not recognize this
architecture", because AutoConfig/AutoModel have no idea what its
"multi_modality" model type is until DeepSeek-VL's own `deepseek_vl` pip
package (installed from their GitHub repo, see requirements.txt) is imported
— that import registers the custom model classes as a side effect. So
DeepSeek-VL gets its own dedicated load/infer path below
(`_load_deepseek_vl` / `_infer_deepseek_vl`), using their documented
VLChatProcessor + MultiModalityCausalLM usage example directly, rather than
going through the generic trust_remote_code path.

All 3 checkpoints are pre-fetched to the local disk cache (hf_cache volume)
in a background thread at startup — that's the slow multi-GB-over-network
part, decoupled from VRAM loading so a user isn't stuck waiting on a cold
download mid-request. VRAM loading itself stays fully lazy (see
_ensure_loaded below): once a model's files are on disk, loading it into
VRAM is comparatively quick (seconds, not minutes), whether that's the first
use or a later switch back to it.

NOT verified end-to-end beyond the one confirmed error above — the
DeepSeek-VL fix and this whole lazy-load/swap mechanism are written directly
against each library's documented API but haven't themselves been run on the
GPU host yet.

BLIP-2 in particular is a weaker instruction-follower than the others (it's
an older captioning/VQA model, not a modern chat-tuned VLM) — the response
parser below falls back to substring class matching if it doesn't return
clean JSON.
"""

import io
import json
import re
import threading

import torch
from fastapi import FastAPI, File, Form, UploadFile
from huggingface_hub import snapshot_download
from PIL import Image

CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]

# Keep in sync with real_inference.py's VLM_MODEL_IDS values — duplicated
# rather than shared since these run in separate containers.
PRELOAD_MODEL_IDS = [
    "OpenGVLab/InternVL2_5-8B",
    "deepseek-ai/deepseek-vl-7b-chat",
    "Salesforce/blip2-opt-2.7b",
]

app = FastAPI()
_state = {"model_id": None, "family": None, "model": None, "processor": None, "tokenizer": None}
_lock = threading.Lock()


def _preload_weights_background() -> None:
    """Downloads (only — no model instantiation, no VRAM) every VLM's
    checkpoint to the local disk cache, one at a time, so switching VLMs
    later never has to fall back to a cold multi-GB download mid-request."""
    for model_id in PRELOAD_MODEL_IDS:
        try:
            print(f"[vlm_server] preloading {model_id} to disk cache...", flush=True)
            snapshot_download(repo_id=model_id)
            print(f"[vlm_server] preload done: {model_id}", flush=True)
        except Exception as exc:
            print(f"[vlm_server] preload FAILED for {model_id}: {exc} "
                  f"(will fall back to downloading on first use instead)", flush=True)


@app.on_event("startup")
def start_preload() -> None:
    threading.Thread(target=_preload_weights_background, daemon=True).start()


def _load_blip2(model_id: str):
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    processor = Blip2Processor.from_pretrained(model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_id, load_in_8bit=True, device_map="auto"
    )
    return {"family": "blip2", "model": model, "processor": processor}


def _load_generic_trust_remote_code(model_id: str):
    """InternVL2.5 — ships its own modeling code on the HF repo and is loaded
    via trust_remote_code, per its model card."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        load_in_8bit=True, device_map="auto",
    ).eval()
    return {"family": "internvl", "model": model, "tokenizer": tokenizer}


def _load_deepseek_vl(model_id: str):
    """DeepSeek-VL (v1) — needs its own deepseek_vl package (see
    requirements.txt), not the generic trust_remote_code path. Follows their
    published usage example (VLChatProcessor + MultiModalityCausalLM via
    AutoModelForCausalLM), with one deviation: quantize only the LLM
    backbone (llm_int8_skip_modules), not the whole model.

    Tried plain full-bf16 first (matching their example exactly): the
    container connection dropped mid-request with no HTTP response at all on
    the lab GPU host, consistent with an OOM kill (7B params in bf16 is
    ~14GB, tight alongside YOLO's own footprint on a 20GB card). Tried
    whole-model load_in_8bit=True next (matching the other two loaders):
    that failed differently and more informatively — a real ValueError deep
    in their custom SAM-based vision tower (deepseek_vl/models/sam.py's
    attention block), because bitsandbytes' int8 matmul kernel only accepts
    2D/3D tensors and the vision tower's forward pass uses higher-rank ones.
    So: quantize just the LLM decoder (the overwhelming majority of the 7B
    params) via llm_int8_skip_modules=["vision_model", "aligner"] (their
    actual top-level submodule names, per the traceback:
    self.aligner(self.vision_model(images))), leaving the vision tower in
    full precision where it's incompatible with 8-bit anyway."""
    from deepseek_vl.models import VLChatProcessor  # noqa: F401 (import registers MultiModalityCausalLM)
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    processor = VLChatProcessor.from_pretrained(model_id)
    quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=["vision_model", "aligner"])
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.bfloat16,
        quantization_config=quant_config, device_map="auto",
    ).eval()
    return {"family": "deepseek_vl", "model": model, "processor": processor, "tokenizer": processor.tokenizer}


def _unload_current() -> None:
    if _state["model"] is None:
        return
    print(f"[vlm_server] unloading {_state['model_id']}", flush=True)
    _state["model"] = None
    _state["processor"] = None
    _state["tokenizer"] = None
    _state["family"] = None
    _state["model_id"] = None
    torch.cuda.empty_cache()


def _ensure_loaded(model_id: str) -> None:
    """Caller must hold _lock. No-op if model_id is already the one loaded —
    switching models (or loading the first one) unloads whatever's current
    first, so at most one VLM is ever resident in VRAM."""
    if _state["model_id"] == model_id:
        return

    _unload_current()

    model_id_lower = model_id.lower()
    print(f"[vlm_server] loading {model_id} ...", flush=True)
    if "blip" in model_id_lower:
        loaded = _load_blip2(model_id)
    elif "deepseek" in model_id_lower:
        loaded = _load_deepseek_vl(model_id)
    else:
        loaded = _load_generic_trust_remote_code(model_id)
    _state.update(loaded)
    _state["model_id"] = model_id
    print(f"[vlm_server] loaded {model_id} as family={_state['family']}", flush=True)


def _build_prompt(user_prompt: str) -> str:
    return user_prompt or (
        f"Classify the object in this image as one of: {', '.join(CLASSES)}. "
        'Respond with JSON: {"class_name": "...", "confidence": 0.0, "reasoning": "..."}'
    )


def _infer_blip2(image: Image.Image, prompt: str) -> str:
    """generate() on BLIP-2's OPT backbone (decoder-only) returns the full
    sequence — input prompt tokens + generated continuation — not just the
    new text. Un-sliced, the decoded text always contains the prompt itself,
    and since our prompt lists "Airplane" first among the class names, the
    lenient fallback in _parse_response matched it every single time
    regardless of what the model actually predicted. Slice off the prompt
    length so only the newly generated tokens get decoded."""
    model, processor = _state["model"], _state["processor"]
    inputs = processor(image, prompt, return_tensors="pt").to(model.device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=200)
    generated_only = out[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_only[0], skip_special_tokens=True)


def _infer_deepseek_vl(image: Image.Image, prompt: str) -> str:
    """Follows DeepSeek-VL's published usage example: a conversation dict
    with an <image_placeholder> token, VLChatProcessor to prepare inputs,
    prepare_inputs_embeds, then generate on model.language_model directly."""
    processor, model, tokenizer = _state["processor"], _state["model"], _state["tokenizer"]

    conversation = [
        {"role": "User", "content": f"<image_placeholder>{prompt}", "images": ["input"]},
        {"role": "Assistant", "content": ""},
    ]
    prepare_inputs = processor(
        conversations=conversation, images=[image.convert("RGB")], force_batchify=True,
    ).to(model.device)

    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=200,
        do_sample=False,
        use_cache=True,
    )
    return tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)


def _infer_generic_chat(image: Image.Image, prompt: str) -> str:
    """InternVL2.5 — expose a `.chat()` method taking a tokenizer, pixel
    values, and a text prompt in its published usage example. Image
    preprocessing here is a simple resize+normalize, not the model's full
    official pipeline (e.g. InternVL's dynamic tiling) — adequate to produce
    a result, may not match published accuracy."""
    from torchvision import transforms

    model, tokenizer = _state["model"], _state["tokenizer"]
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(model.device, torch.bfloat16)
    print(f"[DEBUG] model={type(model)} tokenizer={type(tokenizer)} tokenizer_value={tokenizer!r} "
          f"pixel_values={type(pixel_values)} prompt={type(prompt)} "
          f"chat_attr={type(getattr(model, 'chat', None))}", flush=True)
    response = model.chat(
        tokenizer=tokenizer, pixel_values=pixel_values, question=prompt,
        generation_config=dict(max_new_tokens=200, do_sample=False),
    )
    return response


def _parse_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        data = json.loads(text)
        class_name = data["class_name"]
        if class_name in CLASSES:
            return {
                "class_name": class_name,
                "confidence": float(data.get("confidence", 0.6)),
                "reasoning": str(data.get("reasoning", text)),
            }
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        pass

    # Lenient fallback: find a class name mentioned anywhere in the raw text.
    for cls in CLASSES:
        if re.search(rf"\b{re.escape(cls)}\b", text, re.IGNORECASE):
            return {"class_name": cls, "confidence": 0.5, "reasoning": text}

    raise ValueError(f"Could not extract a known class from model output: {text!r}")


@app.get("/health")
def health():
    return {"status": "ok", "loaded_model_id": _state["model_id"], "family": _state["family"]}


@app.post("/classify")
async def classify(image: UploadFile = File(...), prompt: str = Form(default=""), model_id: str = Form(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    full_prompt = _build_prompt(prompt)

    with _lock:
        _ensure_loaded(model_id)

        if _state["family"] == "blip2":
            raw = _infer_blip2(img, full_prompt)
        elif _state["family"] == "deepseek_vl":
            raw = _infer_deepseek_vl(img, full_prompt)
        else:
            raw = _infer_generic_chat(img, full_prompt)

    return _parse_response(raw)
