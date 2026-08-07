"""
Generic self-hosted VLM classification server. The same image/container is
reused for InternVL2.5-8B, DeepSeek-VL-7B-Chat, and BLIP-2 in
docker-compose.yml's "gpu" profile — MODEL_ID picks which one loads at
startup.

NOT verified end-to-end: written on this machine (no GPU) against each
model's documented loading pattern. BLIP-2's `transformers` API is stable and
should just work. InternVL2.5 and DeepSeek-VL are loaded via
`trust_remote_code=True`, which is how their HF model cards document it, but
that code path can only really be confirmed on the GPU host — see SETUP.md.
Also: DeepSeek-VL (v1, deepseek-ai/deepseek-vl-7b-chat) has historically
shipped its own `deepseek_vl` pip package with a dedicated `VLChatProcessor`
in its official usage examples, rather than working purely through generic
`AutoModel`/`AutoTokenizer` + `trust_remote_code=True` like this file assumes
— if loading or `.chat()` fails for it, that's the first thing to check
against the model's actual HF page.

BLIP-2 in particular is a weaker instruction-follower than the others (it's
an older captioning/VQA model, not a modern chat-tuned VLM) — the response
parser below falls back to substring class matching if it doesn't return
clean JSON.
"""

import io
import json
import os
import re

import torch
from fastapi import FastAPI, File, Form, UploadFile
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "Salesforce/blip2-opt-2.7b")
CLASSES = ["Airplane", "Bird", "Drone", "Helicopter"]

app = FastAPI()
_state = {"family": None, "model": None, "processor": None, "tokenizer": None}


def _load_blip2():
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    processor = Blip2Processor.from_pretrained(MODEL_ID)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_ID, load_in_8bit=True, device_map="auto"
    )
    return {"family": "blip2", "model": model, "processor": processor}


def _load_generic_trust_remote_code():
    """InternVL2.5 / DeepSeek-VL — both ship their own modeling code on the HF
    repo and are loaded via trust_remote_code, per their model cards."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16,
        load_in_8bit=True, device_map="auto",
    ).eval()
    family = "deepseek_vl" if "deepseek" in MODEL_ID.lower() else "internvl"
    return {"family": family, "model": model, "tokenizer": tokenizer}


@app.on_event("startup")
def load_model():
    model_id_lower = MODEL_ID.lower()
    if "blip" in model_id_lower:
        loaded = _load_blip2()
    else:
        loaded = _load_generic_trust_remote_code()
    _state.update(loaded)
    print(f"[vlm_server] loaded {MODEL_ID} as family={_state['family']}")


def _build_prompt(user_prompt: str) -> str:
    return user_prompt or (
        f"Classify the object in this image as one of: {', '.join(CLASSES)}. "
        'Respond with JSON: {"class_name": "...", "confidence": 0.0, "reasoning": "..."}'
    )


def _infer_blip2(image: Image.Image, prompt: str) -> str:
    model, processor = _state["model"], _state["processor"]
    inputs = processor(image, prompt, return_tensors="pt").to(model.device, torch.float16)
    out = model.generate(**inputs, max_new_tokens=200)
    return processor.decode(out[0], skip_special_tokens=True)


def _infer_generic_chat(image: Image.Image, prompt: str) -> str:
    """Shared for InternVL2.5 / DeepSeek-VL — both expose a `.chat()` method
    taking a tokenizer, pixel values, and a text prompt in their published
    usage examples. Image preprocessing here is a simple resize+normalize,
    not each model's full official pipeline (e.g. InternVL's dynamic tiling)
    — adequate to produce a result, may not match published accuracy."""
    from torchvision import transforms

    model, tokenizer = _state["model"], _state["tokenizer"]
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to(model.device, torch.bfloat16)
    response = model.chat(
        tokenizer, pixel_values, prompt,
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
    return {"status": "ok", "model_id": MODEL_ID, "family": _state["family"]}


@app.post("/classify")
async def classify(image: UploadFile = File(...), prompt: str = Form(default="")):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    full_prompt = _build_prompt(prompt)

    if _state["family"] == "blip2":
        raw = _infer_blip2(img, full_prompt)
    else:
        raw = _infer_generic_chat(img, full_prompt)

    return _parse_response(raw)
