"""
Offline language embedding computation.

Pipeline (as per project plan):
    text -> LLM (Flax) -> hidden states -> mean pool -> z

Supports any HuggingFace model name. The Flax backend is tried first
(FlaxLlamaModel, FlaxMistralModel, FlaxAutoModel etc.); falls back to
PyTorch for any model without a Flax implementation.

Model architecture handling:
    Encoder models (BERT, DistilBERT, RoBERTa):
        -> mean pool over all non-padding token hidden states
    Decoder LLMs (GPT-2, GPT-Neo, LLaMA, Mistral, OPT, Falcon):
        -> last non-padding token hidden state
    Encoder-decoder (T5, BART):
        -> mean pool over encoder hidden states

z_embed_dim is auto-detected from the loaded model's hidden_dim and returned
alongside the embedding dict so train.py can wire it into Config.

Upgrade notes:
    - Set llm_model_name = "meta-llama/Llama-2-7b-hf" (needs HF token + GPU)
    - Set llm_model_name = "mistralai/Mistral-7B-v0.1" (needs HF token + GPU)
    - Set llm_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" (no token, ~2GB)
"""

from typing import Dict, List, Tuple

import numpy as np


# ── PDE text descriptions ─────────────────────────────────────────────────────
# Structured descriptions fed to the LLM to produce the conditioning vector z.
# Richer descriptions encode more physics prior — this is the "CoT-style
# prompting" in Phase 4 of the project plan: influencing latent representations
# through prompt design without generating explicit CoT output.

PDE_DESCRIPTIONS: Dict[str, str] = {
    "phy_burgers_sc": (
        "Burgers equation in single-channel mode, periodic boundary conditions, "
        "1D spatial domain, domain extent 1.0, timestep 0.1. "
        "Nonlinear convection scale 0.125, diffusion coefficient 0.0003. "
        "Balances shock steepening against viscous smoothing: "
        "u_t + 0.125 * u * u_x = 0.0003 * u_xx."
    ),
    "phy_diff": (
        "Diffusion equation with diffusion coefficient 0.008, purely dissipative "
        "dynamics, periodic boundary conditions, 1D spatial domain, "
        "domain extent 1.0, timestep 0.1. "
        "Smooths spatial gradients over time: u_t = 0.008 * u_xx."
    ),
    "phy_adv": (
        "Advection equation with advection speed 0.25, pure rightward transport, "
        "periodic boundary conditions, 1D spatial domain, "
        "domain extent 1.0, timestep 0.1. "
        "Translates the initial profile without distortion: u_t + 0.25 * u_x = 0."
    ),
    "phy_ks": (
        "Kuramoto-Sivashinsky equation in combustion (non-conservative) format, "
        "chaotic spatio-temporal dynamics, periodic boundary conditions, "
        "1D spatial domain, domain extent 60.0, timestep 0.1. "
        "Gradient norm nonlinearity scale 1.0, second-order (destabilizing) "
        "coefficient 1.0, fourth-order (stabilizing) coefficient 1.0: "
        "u_t + 0.5 * (u_x)^2 + u_xx + u_xxxx = 0."
    ),
    "phy_burgers": (
        "Burgers equation in conservative multi-channel convection format, "
        "convection scale 0.125, diffusion coefficient 0.0003, "
        "periodic boundary conditions, 1D spatial domain, "
        "domain extent 1.0, timestep 0.1: "
        "u_t + 0.125 * u * u_x = 0.0003 * u_xx."
    ),
    "phy_adv_diff": (
        "Advection-diffusion equation, advection speed 0.25, "
        "diffusion coefficient 0.008, periodic boundary conditions, "
        "1D spatial domain, domain extent 1.0, timestep 0.1: "
        "u_t + 0.25 * u_x = 0.008 * u_xx."
    ),
    "phy_gs": (
        "Gray-Scott reaction-diffusion system with two coupled species u and v, "
        "feed rate 0.04, kill rate 0.06, "
        "diffusivity of species u 2e-5, diffusivity of species v 1e-5, "
        "periodic boundary conditions, 2D spatial domain, "
        "domain extent 1.0, timestep 10.0. "
        "Forms self-organizing Turing patterns via coupled reaction and diffusion."
    ),
    "phy_ks_cons": (
        "Kuramoto-Sivashinsky equation in conservative (fluid dynamics) format, "
        "chaotic spatio-temporal dynamics, periodic boundary conditions, "
        "1D spatial domain, domain extent 60.0. "
        "Convection scale 3.6, second-order destabilizing coefficient 1.44, "
        "fourth-order stabilizing coefficient 0.4: "
        "u_t + 3.6 * u * u_x + 1.44 * u_xx + 0.4 * u_xxxx = 0."
    ),
    "phy_kdv": (
        "Korteweg-de Vries equation with nonlinear convection scale 6.0, "
        "dispersion coefficient 1.0, and numerical stabilization via "
        "hyper-diffusion coefficient 0.125, periodic boundary conditions, "
        "1D spatial domain, domain extent 50.0: "
        "u_t + 6.0 * u * u_x + u_xxx + 0.125 * u_xxxx = 0."
    ),
    "phy_fisher": (
        "Fisher-KPP reaction-diffusion equation with logistic growth dynamics, "
        "periodic boundary conditions, 1D spatial domain, "
        "domain extent 1.0, timestep 0.001. "
        "Diffusion coefficient 0.004, linear growth rate 20.0, "
        "quadratic decay coefficient 20.0: "
        "u_t = 0.004 * u_xx + 20.0 * u * (1 - u)."
    ),
}


# ── Architecture classification ───────────────────────────────────────────────

_ENCODER_TYPES = {
    "bert", "distilbert", "roberta", "albert", "electra",
    "deberta", "xlm_roberta", "camembert",
}
_ENCODER_DECODER_TYPES = {
    "t5", "bart", "pegasus", "mbart", "longt5",
}
# Everything else is treated as decoder


def _classify_arch(model_type: str) -> str:
    """Return "encoder", "encoder_decoder", or "decoder"."""
    t = model_type.lower().replace("-", "_")
    if any(k in t for k in _ENCODER_DECODER_TYPES):
        return "encoder_decoder"
    if any(k in t for k in _ENCODER_TYPES):
        return "encoder"
    return "decoder"


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(model_name: str):
    """
    Load tokenizer + model from HuggingFace.

    Tries the Flax backend first (FlaxAutoModel); falls back to PyTorch.
    Returns (tokenizer, model, arch, hidden_dim, backend).
    """
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(model_name)
    arch = _classify_arch(getattr(config, "model_type", ""))

    # Resolve hidden dimension across different config field names
    hidden_dim = (
        getattr(config, "hidden_size", None)
        or getattr(config, "d_model", None)
        or getattr(config, "n_embd", None)
    )
    if hidden_dim is None:
        raise ValueError(
            f"Cannot determine hidden_dim for '{model_name}'. "
            "Inspect the model config and add the attribute name."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Decoder tokenizers often lack a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Try Flax backend ──────────────────────────────────────────────────────
    backend = None
    model = None
    try:
        from transformers import FlaxAutoModel
        if arch == "encoder_decoder":
            # Use encoder side only
            from transformers import FlaxAutoModelForSeq2SeqLM
            model = FlaxAutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            model = FlaxAutoModel.from_pretrained(model_name)
        backend = "flax"
    except Exception as flax_err:
        pass

    # ── Fall back to PyTorch ──────────────────────────────────────────────────
    if model is None:
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            backend = "torch"
        except Exception as torch_err:
            raise RuntimeError(
                f"Could not load '{model_name}' with either Flax or PyTorch.\n"
                f"  Flax error : {flax_err}\n"
                f"  Torch error: {torch_err}"
            )

    print(f"  Model  : {model_name}")
    print(f"  Arch   : {arch}  |  Backend : {backend}  |  Hidden dim : {hidden_dim}")
    return tokenizer, model, arch, hidden_dim, backend


# ── Embedding extraction ──────────────────────────────────────────────────────

def _mean_pool(hidden, attention_mask):
    """Masked mean pool over sequence dimension. Works for both jax and numpy arrays."""
    import jax.numpy as jnp
    mask = attention_mask.astype(jnp.float32)[:, :, None]  # (1, seq_len, 1)
    return (hidden * mask).sum(axis=1) / mask.sum(axis=1)  # (1, H)


def _embed_flax(text: str, tokenizer, model, arch: str, max_length: int = 128) -> np.ndarray:
    """Extract a 1-D embedding using a Flax model.

    Mean pools over all non-padding token hidden states for all model types.
    For encoder-decoder models, only the encoder side is used.
    """
    inputs = tokenizer(
        text,
        return_tensors="jax",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    if arch == "encoder_decoder":
        encoder_out = model.encode(**{k: v for k, v in inputs.items()})
        hidden = encoder_out.last_hidden_state          # (1, seq_len, H)
    else:
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state              # (1, seq_len, H)

    z = _mean_pool(hidden, inputs["attention_mask"])
    return np.array(z[0], dtype=np.float32)


def _embed_torch(text: str, tokenizer, model, arch: str, max_length: int = 128) -> np.ndarray:
    """Extract a 1-D embedding using a PyTorch model (fallback).

    Mean pools over all non-padding token hidden states for all model types.
    For encoder-decoder models, only the encoder side is used.
    """
    import torch

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    with torch.no_grad():
        if arch == "encoder_decoder":
            hidden = model.encoder(**inputs).last_hidden_state
        else:
            hidden = model(**inputs, output_hidden_states=True).last_hidden_state

        mask = inputs["attention_mask"].float().unsqueeze(-1)   # (1, seq_len, 1)
        z = (hidden * mask).sum(1) / mask.sum(1)                # (1, H)
        return z[0].cpu().numpy().astype(np.float32)


# ── Chain-of-thought generation + caching ────────────────────────────────────

def _build_cot_user_prompt(scenario_key: str, declarative: str) -> str:
    """The user turn we send to TinyLlama-Chat."""
    return (
        "You are analyzing a partial differential equation (PDE) for a "
        "numerical solver. Reason step by step.\n\n"
        f"PDE description: {declarative}\n\n"
        "Produce a concise analysis:\n"
        "Step 1 — Identify the equation and its canonical form.\n"
        "Step 2 — Classify each term (convection / diffusion / dispersion / reaction).\n"
        "Step 3 — Compare the relative scales of those terms.\n"
        "Step 4 — Describe the expected qualitative dynamics.\n"
        "Step 5 — Note the implication for a neural-operator solver on this domain.\n"
        "Keep each step to 1–2 sentences."
    )


def _apply_chat_template(tokenizer, user_prompt: str) -> str:
    """Render the tokenizer's chat template (TinyLlama-Chat ships one)."""
    assert tokenizer.chat_template is not None, (
        "Tokenizer has no chat_template; TinyLlama-Chat-v1.0 ships one. "
        "If you swapped models, add a manual <|system|>/<|user|>/<|assistant|> fallback."
    )
    messages = [
        {"role": "system", "content": "You are an expert in partial differential equations and numerical analysis."},
        {"role": "user",   "content": user_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _cot_cache_key(
    model_name: str,
    scenario_key: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Content-addressed cache key — any input change invalidates the cache."""
    import hashlib
    payload = "\x1f".join([
        model_name, scenario_key, user_prompt,
        str(max_new_tokens), f"{temperature:.4f}",
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _generate_cot_torch(
    tokenizer,
    user_prompt: str,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> str:
    """Run .generate() under PyTorch (Flax Llama has no causal LM head).
    Greedy when temperature == 0.0.
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    lm.to(device)
    lm.eval()

    prompt_text = _apply_chat_template(tokenizer, user_prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    do_sample = temperature > 0.0
    with torch.no_grad():
        out = lm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else 1.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    reply_ids = out[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()
    del lm
    if device == "cuda":
        torch.cuda.empty_cache()
    return text


def _resolve_cot_text(
    scenario_key: str,
    declarative: str,
    cfg,
    tokenizer,
) -> str:
    """Return CoT text for a scenario — from disk cache if present, else generate + persist."""
    import json
    import pathlib

    user_prompt = _build_cot_user_prompt(scenario_key, declarative)
    cache_dir = pathlib.Path(cfg.cot_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cot_cache_key(
        cfg.llm_model_name, scenario_key, user_prompt,
        cfg.cot_gen_max_new_tokens, cfg.cot_gen_temperature,
    )
    path = cache_dir / f"{scenario_key}_{key}.json"
    if path.exists():
        print(f"  CoT cache hit  : {path}")
        return json.loads(path.read_text())["cot_text"]

    print(f"  CoT generating : {scenario_key} (greedy, max_new_tokens={cfg.cot_gen_max_new_tokens})")
    text = _generate_cot_torch(
        tokenizer, user_prompt, cfg.llm_model_name,
        cfg.cot_gen_max_new_tokens, cfg.cot_gen_temperature, cfg.cot_gen_seed,
    )
    record = {
        "scenario_key":   scenario_key,
        "model_name":     cfg.llm_model_name,
        "user_prompt":    user_prompt,
        "cot_text":       text,
        "max_new_tokens": cfg.cot_gen_max_new_tokens,
        "temperature":    cfg.cot_gen_temperature,
        "seed":           cfg.cot_gen_seed,
        "cache_key":      key,
    }
    path.write_text(json.dumps(record, indent=2))
    print(f"  CoT cached     : {path}")
    return text


# ── Public API ────────────────────────────────────────────────────────────────

def precompute_embeddings(
    scenario_keys: List[str],
    model_name: str,
    cfg=None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Precompute language embeddings for all requested PDE scenarios.

    Loads `model_name` once (Flax if supported, PyTorch otherwise).
    Runs one forward pass per PDE description.

    Selection of the text to embed is controlled by `cfg.prompt_style`:
        - "declarative"      : PDE_DESCRIPTIONS[key], max_length=128 (original)
        - "declarative_long" : PDE_DESCRIPTIONS[key], max_length=cfg.cot_max_length
        - "cot_generated"    : TinyLlama-generated reasoning (cached), max_length=cfg.cot_max_length

    Returns:
        embeddings : Dict[scenario_key -> z]
                     z is float32 numpy, shape (hidden_dim,), unit-normalized
        embed_dim  : int — the model's hidden dimension (set this as Config.z_embed_dim)
    """
    tokenizer, model, arch, hidden_dim, backend = _load_model(model_name)
    embed_fn = _embed_flax if backend == "flax" else _embed_torch

    style = getattr(cfg, "prompt_style", "declarative") if cfg is not None else "declarative"
    if style == "declarative":
        max_len = 128
    else:
        max_len = getattr(cfg, "cot_max_length", 512)

    # ── Resolve the string to embed for each scenario ───────────────────────
    texts: Dict[str, str] = {}
    for key in scenario_keys:
        declarative = PDE_DESCRIPTIONS.get(key, key)
        if style in ("declarative", "declarative_long"):
            texts[key] = declarative
        elif style == "cot_generated":
            texts[key] = _resolve_cot_text(key, declarative, cfg, tokenizer)
        else:
            raise ValueError(
                f"Unknown prompt_style: {style!r}. "
                "Expected one of: 'declarative', 'declarative_long', 'cot_generated'."
            )

    # ── Embed each text, unit-normalize, collect ────────────────────────────
    embeddings: Dict[str, np.ndarray] = {}
    for key in scenario_keys:
        z = embed_fn(texts[key], tokenizer, model, arch, max_length=max_len)
        z = z / (np.linalg.norm(z) + 1e-8)
        embeddings[key] = z
        print(
            f"  z[{key}]  style={style}  shape={z.shape}  "
            f"text_chars={len(texts[key])}  norm={np.linalg.norm(z):.4f}"
        )

    return embeddings, hidden_dim
