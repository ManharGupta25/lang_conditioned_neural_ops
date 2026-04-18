"""
precompute_embeddings.py — Precompute and save LLM embeddings to disk.

Use this as a workaround when PyTorch and JAX conflict over CUDA/cuBLAS
versions (e.g. CUDA 12.9 with JAX built against CUDA 12.6).

Workflow:
    1. python3 precompute_embeddings.py      # PyTorch only, saves .npy files
    2. python3 train.py --phase 2 --load-embeddings  # JAX only, loads from disk

Embeddings are saved to:
    <embeddings_dir>/<llm_short_name>/<scenario_key>.npy
    <embeddings_dir>/<llm_short_name>/embed_dim.txt

Normal workflow (no CUDA conflict) remains unchanged:
    python3 train.py --phase 2              # computes embeddings inline as before
"""

import pathlib
import numpy as np
from config import Config
from data.embeddings import precompute_embeddings


def main():
    cfg = Config()
    embeddings_dir = pathlib.Path(cfg.embeddings_dir) / cfg.llm_short_name()
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Precomputing language embeddings (PyTorch only)")
    print(f"LLM : {cfg.llm_model_name}")
    print(f"PDEs: {cfg.pde_scenarios}")
    print(f"Saving to: {embeddings_dir}/")
    print("=" * 65)

    embeddings, embed_dim = precompute_embeddings(cfg.pde_scenarios, cfg.llm_model_name)

    for key, z in embeddings.items():
        path = embeddings_dir / f"{key}.npy"
        np.save(path, z)
        print(f"  Saved: {path}  shape={z.shape}  norm={np.linalg.norm(z):.4f}")

    # Save embed_dim so train.py can read it without reloading the model
    (embeddings_dir / "embed_dim.txt").write_text(str(embed_dim))
    print(f"\nEmbed dim: {embed_dim} -> {embeddings_dir}/embed_dim.txt")
    print(f"\nDone. Run  python3 train.py --phase 2  to train (embeddings will be loaded from disk automatically).")


if __name__ == "__main__":
    main()
