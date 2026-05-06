"""
precompute_embeddings.py — Precompute and save LLM embeddings to disk.

Use this as a workaround when PyTorch and JAX conflict over CUDA/cuBLAS
versions (e.g. CUDA 12.9 with JAX built against CUDA 12.6).

Workflow:
    1. python3 precompute_embeddings.py      # PyTorch only, saves .npy files
    2. python3 train.py --phase 2 --load-embeddings  # JAX only, loads from disk

Embeddings are saved to a backend+style+layer_mode-keyed subdirectory:
    <embeddings_dir>/<llm_short_name>/<backend>_<style>_<layer_mode>/<key>.npy
    <embeddings_dir>/<llm_short_name>/<backend>_<style>_<layer_mode>/embed_dim.txt

Exception: automodel+declarative+last (all defaults) use the root directory
for backwards compatibility with previously precomputed files:
    <embeddings_dir>/<llm_short_name>/<scenario_key>.npy

Normal workflow (no CUDA conflict) remains unchanged:
    python3 train.py --phase 2              # computes embeddings inline as before
"""

import pathlib
import numpy as np
from config import Config
from data.embeddings import precompute_embeddings


def main():
    cfg = Config()
    base_dir = pathlib.Path(cfg.embeddings_dir) / cfg.llm_short_name()

    style   = cfg.prompt_style
    layer_mode = cfg.embedding_layer_mode
    backend = cfg.embedding_backend

    # Mirror the cache-key logic in precompute_embeddings() exactly so that
    # saved files are found by the cache lookup in train.py.
    if backend == "automodel" and style == "declarative" and layer_mode == "last":
        save_dir = base_dir          # legacy flat layout, no subdirectory
    else:
        save_dir = base_dir / f"{backend}_{style}_{layer_mode}"

    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Precomputing language embeddings (PyTorch only)")
    print(f"LLM        : {cfg.llm_model_name}")
    print(f"PDEs       : {cfg.pde_scenarios}")
    print(f"Backend    : {backend}  |  Style: {style}  |  Layer mode: {layer_mode}")
    print(f"Saving to  : {save_dir}/")
    print("=" * 65)

    embeddings, embed_dim = precompute_embeddings(cfg.pde_scenarios, cfg.llm_model_name, cfg=cfg)

    for key, z in embeddings.items():
        path = save_dir / f"{key}.npy"
        np.save(path, z)
        print(f"  Saved: {path}  shape={z.shape}  norm={np.linalg.norm(z):.4f}")

    (save_dir / "embed_dim.txt").write_text(str(embed_dim))
    print(f"\nEmbed dim: {embed_dim} -> {save_dir}/embed_dim.txt")
    print(f"\nDone. Run  python3 train.py --phase 2  to train (embeddings loaded from disk).")


if __name__ == "__main__":
    main()
