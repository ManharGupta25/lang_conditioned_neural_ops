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


def _embed_flax(text: str, tokenizer, model, arch: str) -> np.ndarray:
    """Extract a 1-D embedding using a Flax model.

    Mean pools over all non-padding token hidden states for all model types.
    For encoder-decoder models, only the encoder side is used.
    """
    inputs = tokenizer(
        text,
        return_tensors="jax",
        padding=True,
        truncation=True,
        max_length=128,
    )

    if arch == "encoder_decoder":
        encoder_out = model.encode(**{k: v for k, v in inputs.items()})
        hidden = encoder_out.last_hidden_state          # (1, seq_len, H)
    else:
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.last_hidden_state              # (1, seq_len, H)

    z = _mean_pool(hidden, inputs["attention_mask"])
    return np.array(z[0], dtype=np.float32)


def _embed_torch(text: str, tokenizer, model, arch: str) -> np.ndarray:
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
        max_length=128,
    )

    with torch.no_grad():
        if arch == "encoder_decoder":
            hidden = model.encoder(**inputs).last_hidden_state
        else:
            hidden = model(**inputs, output_hidden_states=True).last_hidden_state

        mask = inputs["attention_mask"].float().unsqueeze(-1)   # (1, seq_len, 1)
        z = (hidden * mask).sum(1) / mask.sum(1)                # (1, H)
        return z[0].cpu().numpy().astype(np.float32)


# ── Public API ────────────────────────────────────────────────────────────────

def precompute_embeddings(
    scenario_keys: List[str],
    model_name: str,
    embeddings_dir: str = None,
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Precompute language embeddings for all requested PDE scenarios.

    If embeddings_dir is set and all .npy files + embed_dim.txt exist there,
    loads from disk instead of running the LLM — useful when PyTorch and JAX
    conflict over CUDA versions. Otherwise computes inline as normal.

    Returns:
        embeddings : Dict[scenario_key -> z]
                     z is float32 numpy, shape (hidden_dim,), unit-normalized
        embed_dim  : int — the model's hidden dimension
    """
    import pathlib

    # ── Try loading from disk first ───────────────────────────────────────────
    if embeddings_dir is not None:
        cache_dir = pathlib.Path(embeddings_dir)
        dim_file = cache_dir / "embed_dim.txt"
        all_exist = dim_file.exists() and all(
            (cache_dir / f"{key}.npy").exists() for key in scenario_keys
        )
        if all_exist:
            print(f"  Loading precomputed embeddings from {cache_dir}/")
            embed_dim = int(dim_file.read_text().strip())
            embeddings: Dict[str, np.ndarray] = {}
            for key in scenario_keys:
                z = np.load(cache_dir / f"{key}.npy")
                embeddings[key] = z
                print(f"  z[{key}]  shape={z.shape}  norm={np.linalg.norm(z):.4f}")
            return embeddings, embed_dim

    # ── Compute inline ────────────────────────────────────────────────────────
    tokenizer, model, arch, hidden_dim, backend = _load_model(model_name)
    embed_fn = _embed_flax if backend == "flax" else _embed_torch

    embeddings = {}
    for key in scenario_keys:
        text = PDE_DESCRIPTIONS.get(key, key)
        z = embed_fn(text, tokenizer, model, arch)
        z = z / (np.linalg.norm(z) + 1e-8)
        embeddings[key] = z
        print(f"  z[{key}]  shape={z.shape}  norm={np.linalg.norm(z):.4f}")

    return embeddings, hidden_dim
