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

from config import Config
from data.embeddings import _build_cot_user_prompt, _resolve_cot_text
from transformers import AutoTokenizer


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

    from data.embeddings import PDE_DESCRIPTIONS
    for key in cfg.pde_scenarios:
        declarative = PDE_DESCRIPTIONS.get(key, key)
        text = _resolve_cot_text(key, declarative, cfg, tokenizer)
        print(f"\n── {key} ──")
        print(text[:400] + ("..." if len(text) > 400 else ""))
        print()


if __name__ == "__main__":
    main()
