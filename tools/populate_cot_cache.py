#!/usr/bin/env python
"""Pre-populate the TinyLlama CoT cache for all configured PDE scenarios.

Run this once before submitting any cot_generated sbatch jobs so that the
parallel training jobs read from cache rather than racing on generation.

    python tools/populate_cot_cache.py

Writes JSON files to cfg.cot_cache_dir (default: cache/cot_generated/).
Idempotent — skips any cache entries that already exist.
"""

import pathlib
import sys

# Allow running from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import Config
from data.embeddings import (
    PDE_DESCRIPTIONS,
    _resolve_cot_text,
    _build_cot_user_prompt,
    _cot_cache_key,
)


def main() -> None:
    cfg = Config()
    cfg.prompt_style = "cot_generated"

    print(f"Populating CoT cache under {cfg.cot_cache_dir}/")
    print(f"  Model  : {cfg.llm_model_name}")
    print(f"  PDEs   : {cfg.pde_scenarios}")
    print(f"  Greedy : temperature={cfg.cot_gen_temperature}  max_new_tokens={cfg.cot_gen_max_new_tokens}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check which scenarios actually need generation (not yet cached).
    cache_dir = pathlib.Path(cfg.cot_cache_dir)
    needs_gen = []
    for key in cfg.pde_scenarios:
        declarative = PDE_DESCRIPTIONS.get(key, key)
        user_prompt = _build_cot_user_prompt(key, declarative)
        cache_key = _cot_cache_key(
            cfg.llm_model_name, key, user_prompt,
            cfg.cot_gen_max_new_tokens, cfg.cot_gen_temperature,
        )
        if not (cache_dir / f"{key}_{cache_key}.json").exists():
            needs_gen.append(key)

    # Load the generation LLM once only if at least one scenario needs it.
    lm_gen = None
    if needs_gen:
        print(f"  {len(needs_gen)} scenario(s) need generation; loading {cfg.llm_model_name} ...")
        lm_gen = AutoModelForCausalLM.from_pretrained(
            cfg.llm_model_name, dtype=torch.float32, device_map="auto"
        )
        lm_gen.eval()
    else:
        print("  All scenarios already cached — nothing to generate.")

    for key in cfg.pde_scenarios:
        declarative = PDE_DESCRIPTIONS.get(key, key)
        text = _resolve_cot_text(key, declarative, cfg, tokenizer, lm=lm_gen)
        print(f"\n── {key} ──")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()

    if lm_gen is not None:
        del lm_gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
