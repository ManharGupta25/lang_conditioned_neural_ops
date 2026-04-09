# Language-Conditioned Neural Operators

> **CS288 Project** — Can language serve as a general-purpose conditioning interface for PDE solvers?

Core idea: extract a latent vector **z** from an LLM's hidden states and use it to modulate a Fourier Neural Operator (FNO), enabling language to act as a **latent control signal** for neural PDE solvers. Training is fully differentiable (no RL).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Architecture Overview](#architecture-overview)
3. [Detailed Architecture](#detailed-architecture)
   - [Language Embedding Pipeline](#1-language-embedding-pipeline)
   - [ConditionedFNO](#2-conditionedfno)
   - [FiLM Conditioning](#3-film-conditioning)
   - [Spectral Gating](#4-spectral-gating)
4. [Training Pipeline](#training-pipeline)
   - [Phase 1 — Baseline FNOs](#phase-1--baseline-fnos)
   - [Phase 2 — Conditioned FNOs](#phase-2--conditioned-fnos-llm-frozen)
5. [Evaluation & Visualization](#evaluation--visualization)
6. [Configuration](#configuration)
7. [PDE Scenarios](#pde-scenarios)
8. [What Has Been Implemented](#what-has-been-implemented)
9. [What Still Needs to Be Done](#what-still-needs-to-be-done)
10. [How to Run](#how-to-run)

---

## Repository Structure

```
lang_conditioned_neural_ops/
├── config.py                     # All hyperparameters and paths
├── train.py                      # Two-phase training script
├── plot.py                       # Result visualization (3 figure types)
├── requirements.txt              # Dependencies
├── models/
│   ├── __init__.py
│   └── conditioned_fno.py        # ConditionedFNO, FiLMLayer, SpectralGating
├── data/
│   ├── __init__.py
│   └── embeddings.py             # LLM embedding pipeline + PDE descriptions
├── checkpoints/                  # Saved .eqx model checkpoints
├── results/                      # Training metric CSVs
└── figures/                      # Generated comparison plots
```

---

## Architecture Overview

```
                          ┌─────────────────────────┐
                          │   Natural Language PDE   │
                          │       Description        │
                          └────────────┬────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   LLM (frozen weights)   │
                          │  e.g. DistilBERT / GPT2  │
                          └────────────┬────────────┘
                                       │
                                  hidden states
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  Mean Pool + Normalize   │
                          └────────────┬────────────┘
                                       │
                                  z_embed (768-d)
                                       │
                              stop_gradient(z)
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │  Linear Projection       │
                          │  z_embed_dim → cond_dim  │  ← trained
                          └────────────┬────────────┘
                                       │
                                  z_cond (64-d)
                                       │
  u_t ──► [FNO Lifting] ──►┌──────────┴──────────┐
                            │                     │
                            ▼                     │
                    [Spectral Block 0]            │
                            │                     │
                    [Conditioning Layer] ◄────────┤  (FiLM or Spectral Gating)
                            │                     │
                            ▼                     │
                    [Spectral Block 1]            │
                            │                     │
                    [Conditioning Layer] ◄────────┤
                            │                     │
                           ...                   ...
                            │                     │
                            ▼                     │
                    [Spectral Block N-1]          │
                            │                     │
                    [Conditioning Layer] ◄────────┘
                            │
                            ▼
                    [FNO Projection]
                            │
                            ▼
                          u_{t+1}
```

The model is an **autoregressive timestepper**: given the PDE state at time `t`, it predicts the state at time `t+1`. To generate full trajectories, the output is fed back as input repeatedly.

---

## Detailed Architecture

### 1. Language Embedding Pipeline

**File:** `data/embeddings.py`

Each PDE scenario has a hand-written natural language description encoding its physics (equation form, coefficients, boundary conditions, qualitative behavior). For example:

> *"Burgers equation in single-channel mode, periodic boundary conditions, 1D spatial domain, domain extent 1.0, timestep 0.1. Nonlinear convection scale 0.125, diffusion coefficient 0.0003. Balances shock steepening against viscous smoothing: u_t + 0.125 \* u \* u_x = 0.0003 \* u_xx."*

**Embedding extraction steps:**

1. **Load model** — any HuggingFace model. Auto-detects Flax vs PyTorch backend, and classifies architecture as encoder / decoder / encoder-decoder.
2. **Tokenize** — max 128 tokens, padded.
3. **Forward pass** — extract `last_hidden_state` (for encoder-decoder, only the encoder side).
4. **Mean pool** — mask-aware average over non-padding token hidden states: `z = Σ(h_i * mask_i) / Σ(mask_i)`.
5. **Normalize** — `z = z / (‖z‖ + 1e-8)` for numerical stability.

**Output:** one float32 vector per PDE of shape `(hidden_dim,)`. This is precomputed once before training and stored inside the model as a frozen constant.

**Supported LLMs (configurable in `config.py`):**

| Model | Dim | Type | Params |
|---|---|---|---|
| `distilbert-base-uncased` (default) | 768 | encoder | 66M |
| `gpt2` | 768 | decoder | 124M |
| `EleutherAI/gpt-neo-125m` | 768 | decoder | 125M |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 | decoder | 1.1B |
| `facebook/opt-350m` | 512 | decoder | 350M |
| `meta-llama/Llama-2-7b-hf` | 4096 | decoder | 7B |
| `mistralai/Mistral-7B-v0.1` | 4096 | decoder | 7B |

---

### 2. ConditionedFNO

**File:** `models/conditioned_fno.py` — class `ConditionedFNO`

Wraps a `pdequinox.arch.ClassicFNO` by extracting its components (lifting, spectral blocks, projection) and interleaving conditioning layers between spectral blocks.

**Components:**

| Component | Description | Trainable? |
|---|---|---|
| `z` | Frozen LLM embedding `(z_embed_dim,)` | No (`stop_gradient`) |
| `z_proj` | Linear: `z_embed_dim → cond_dim` | Yes |
| `fno_lifting` | Input projection (channels → hidden) | Yes |
| `fno_blocks[0..5]` | 6 spectral convolution blocks | Yes |
| `film_layers[0..5]` | FiLM conditioning (one per block) | Yes (if active) |
| `spectral_gating_layers[0..5]` | Spectral gating (one per block) | Yes (if active) |
| `fno_projection` | Output projection (hidden → channels) | Yes |

**Key design decisions:**

- **Both** conditioning layer types are always instantiated so the PyTree structure is stable for serialization — the inactive method simply receives zero gradients.
- The embedding `z` is frozen via `jax.lax.stop_gradient(self.z)` at the start of every forward pass, preventing any gradient flow into the LLM embedding.
- The FNO trunk is warm-started from Phase 1 baseline weights. Only the conditioning components (`z_proj`, FiLM/spectral layers) start from random initialization.

**Forward pass:**

```python
def __call__(self, u_t):
    z = stop_gradient(self.z)          # freeze embedding
    z_cond = self.z_proj(z)            # project to cond_dim

    x = self.fno_lifting(u_t)          # lift to hidden channels

    for block, cond_layer in zip(self.fno_blocks, self.<active>_layers):
        x = block(x)                   # spectral convolution
        x = cond_layer(x, z_cond)      # apply conditioning

    return self.fno_projection(x)      # project back to output channels
```

---

### 3. FiLM Conditioning

**File:** `models/conditioned_fno.py` — class `FiLMLayer`

Feature-wise Linear Modulation (FiLM) — the **primary baseline** conditioning method.

**Operation:**
```
γ = Linear(z_cond)    →  (hidden_channels,)
β = Linear(z_cond)    →  (hidden_channels,)
x_out = (1 + γ) · x + β
```

- The `(1 + γ)` formulation ensures FiLM starts as an identity transform when weights are near zero at initialization (γ ≈ 0, β ≈ 0 → output ≈ input).
- γ and β are reshaped to `(channels, 1, 1, ...)` to broadcast over all spatial dimensions.
- **Parameters per block:** 2 × (cond_dim × hidden_channels + hidden_channels) = 2 × (64 × 64 + 64) = **8,320**
- **Total FiLM parameters (6 blocks):** ~49,920

---

### 4. Spectral Gating

**File:** `models/conditioned_fno.py` — class `SpectralGating`

Modulates the first `num_modes` Fourier coefficients using a learned, frequency-dependent gate.

**Operation:**
```
x̂ = rfft(x, axis=-1)                      # to Fourier space
gate_raw = MLP(z_cond)                      # (num_modes,)
gate = 1 + α · tanh(gate_raw)              # α = 0.1
x̂[:, :num_modes] *= gate                   # modulate low frequencies
x_out = irfft(x̂)                           # back to spatial
```

- **MLP architecture:** `cond_dim → cond_dim → num_modes` (depth 2, ReLU activations).
- **Identity at init:** MLP weights ≈ 0 → gate_raw ≈ 0 → tanh(0) = 0 → gate ≈ 1 → no modulation.
- `α = 0.1` bounds the gate to `[0.9, 1.1]` initially, keeping training stable.
- Only the first `num_modes` frequency bins are gated; higher frequencies pass through unchanged.
- **Parameters per block:** cond_dim² + cond_dim × num_modes + biases ≈ 64² + 64 × 12 ≈ **4,864**
- **Total spectral gating parameters (6 blocks):** ~29,184

---

## Training Pipeline

### Phase 1 — Baseline FNOs

```bash
python train.py --phase 1
```

Trains a plain FNO (no conditioning) on each PDE scenario using APEBench's built-in training loop.

- APEBench automatically vmaps over `num_seeds` (3) for parallel multi-seed training.
- For each PDE: creates an APEBench scenario → trains FNO → saves checkpoint (`.eqx`) and metrics (`.csv`).
- **Network:** `fno;12;64;6;gelu` — 12 Fourier modes, 64 hidden channels, 6 blocks, GELU activation.
- **Optimizer:** Adam with cosine warmup schedule, peak LR 1e-3, warmup steps = max(200, num_steps/6).
- **Output:** Per-PDE checkpoint at `checkpoints/<dir>/<pde>_baseline.eqx` and CSV at `results/<dir>/<pde>_baseline.csv`.

### Phase 2 — Conditioned FNOs (LLM frozen)

```bash
python train.py --phase 2
```

For each PDE scenario:

1. **Load Phase 1 checkpoint** — deserializes the batched baseline FNO `[num_seeds, ...]`.
2. **Precompute LLM embeddings** — runs the frozen LLM once per PDE description, stores the resulting `z` vectors.
3. **Build warm-started models** — for each seed:
   - Create a `ConditionedFNO` with random conditioning weights.
   - Copy the FNO trunk (lifting, blocks, projection) from the corresponding Phase 1 seed's weights via `eqx.tree_at`.
   - Stack all seeds into a batched model.
4. **Train with vmap** — `eqx.filter_vmap(train_one)(batch, keys)` gives the same GPU parallelism as Phase 1.
5. **Evaluate** — `scenario.perform_tests(trained_batch)` computes nRMSE across the test horizon.
6. **Save** — checkpoint and APEBench-format CSV.

After Phase 2, a summary table is printed comparing baseline vs conditioned final nRMSE for each PDE.

---

## Evaluation & Visualization

```bash
python plot.py
```

Reads all CSVs from `results/` and generates three figures in `figures/`:

| Figure | Description |
|---|---|
| `nrmse_rollout.png` | nRMSE vs test timestep — one subplot per PDE, baseline vs conditioned lines with 95% CI shading across seeds |
| `train_loss.png` | Training MSE loss vs step (log scale) — one subplot per PDE, 95% CI shading |
| `final_nrmse_bar.png` | Grouped bar chart of final-timestep nRMSE across all PDEs and conditions, error bars = std |

**Metric:** Normalized RMSE (nRMSE) — computed by APEBench over 100 test samples across a 200-timestep rollout horizon.

---

## Configuration

All hyperparameters live in `config.py` (`Config` dataclass):

| Parameter | Default | Description |
|---|---|---|
| `fno_modes` | 12 | Number of Fourier modes retained per block |
| `fno_hidden` | 64 | Hidden channel dimension |
| `fno_blocks` | 6 | Number of spectral convolution blocks |
| `fno_activation` | `"gelu"` | Activation function |
| `z_cond_dim` | 64 | Projection dimension for conditioning vector |
| `conditioning_method` | `"film"` | `"film"` or `"spectral_gating"` |
| `llm_model_name` | `"distilbert-base-uncased"` | HuggingFace model for embeddings |
| `num_train_steps` | 20,000 | Training steps per phase |
| `num_train_samples` | 1,000 | Training trajectories |
| `batch_size` | 50 | Batch size |
| `train_temporal_horizon` | 50 | Rollout steps during training |
| `test_temporal_horizon` | 200 | Rollout steps during evaluation |
| `num_seeds` | 3 | Parallel seeds (vmapped) |
| `num_points` | 128 | Spatial resolution |

---

## PDE Scenarios

Seven PDE scenarios from APEBench, covering a range of physics:

| Key | PDE | Dims | Behavior |
|---|---|---|---|
| `phy_burgers_sc` | Burgers (single-channel) | 1D | Nonlinear advection + diffusion, shock formation |
| `phy_diff` | Diffusion | 1D | Purely dissipative, smoothing |
| `phy_adv` | Advection | 1D | Pure transport, shape-preserving |
| `phy_adv_diff` | Advection-Diffusion | 1D | Combined transport + smoothing |
| `phy_ks` | Kuramoto-Sivashinsky | 1D | Chaotic spatiotemporal dynamics |
| `phy_burgers` | Burgers (multi-channel) | 1D | Conservative convection form |
| `phy_gs` | Gray-Scott | 2D | Reaction-diffusion, Turing patterns |

Additional PDE descriptions exist in `data/embeddings.py` for future use: `phy_ks_cons`, `phy_kdv`, `phy_fisher`.

---

## What Has Been Implemented

### Completed (per project plan)

- [x] **Phase 1 — Baseline FNOs:** Training loop, checkpointing, multi-seed vmap, APEBench integration
- [x] **Phase 2 — Language conditioning (LLM frozen):** Full pipeline: text → LLM → mean pool → z → projection → FNO conditioning
- [x] **FiLM conditioning:** Feature-wise linear modulation after each spectral block (primary baseline)
- [x] **Spectral gating:** Per-frequency gate modulation in Fourier space
- [x] **Warm-starting:** Phase 2 models inherit FNO trunk from Phase 1 checkpoints
- [x] **Multi-LLM support:** Architecture-agnostic embedding pipeline (encoder / decoder / encoder-decoder), Flax + PyTorch backends
- [x] **Evaluation pipeline:** nRMSE rollout curves, training loss curves, final metric bar charts with CI/std
- [x] **PDE descriptions:** Rich natural language descriptions for 10 PDEs with equation forms and coefficients
- [x] **APEBench architecture registration:** `make_cfno_constructor` for seamless APEBench integration

---

## What Still Needs to Be Done

### High Priority (must do per project plan)

- [ ] **Structured numerical conditioning baseline** — The project plan marks this as **IMPORTANT**: compare against FNO + structured conditioning (no language). This means feeding PDE parameters (coefficients, etc.) directly as a numerical vector instead of through language. This is the key control to show that language adds value beyond just providing parameter information.

- [ ] **Spectral filter modulation** — The third conditioning method listed on the primary ablation axis. Currently only FiLM and spectral gating are implemented. Spectral modulation would directly modulate the learned spectral filters (weights) of each FNO block rather than gating the output.

- [ ] **Phase 3 — Generalization experiments:** Evaluate trained models on:
  - Parametric generalization (vary PDE parameters outside training range)
  - Superresolution (test at higher spatial resolution than training)
  - Non-periodic boundary conditions (FNO limitation test)

- [ ] **Phase 5 — LoRA fine-tuning:** Enable gradient flow into the LLM via LoRA adapters:
  - Remove `stop_gradient` on `z` and add LoRA to the LLM
  - Co-train: FNO + projection + LoRA adapters (FNO is NOT frozen)
  - Rank ablations and target module ablations (attention vs extended)
  - Must be compatible with Tinker (standard LoRA only, no DoRA)

### Medium Priority (secondary ablations)

- [ ] **Projection layer ablation** — Compare linear projection (current) vs MLP projection for `z_embed_dim → cond_dim`. The infrastructure exists to swap `z_proj` from `eqx.nn.Linear` to `eqx.nn.MLP`.

- [ ] **Cross-PDE generalization** — Train a single language-conditioned FNO on multiple PDEs simultaneously, then test transfer to unseen PDE families. This requires modifying the training loop to sample from multiple scenarios per batch.

- [ ] **Larger LLM experiments** — Currently defaults to DistilBERT (66M params). Run with TinyLlama-1.1B and Llama-2-7B to measure if a stronger language prior improves conditioning quality.

### Lower Priority (later / stretch goals)

- [ ] **Phase 4 — CoT-style prompting** — Use richer / structured prompts (chain-of-thought style) to influence latent representations. The PDE descriptions already encode physics priors; this would involve systematic prompt engineering experiments with the frozen LLM.

- [ ] **LLM pooling variants** — Ablate the embedding extraction strategy:
  - Mean pooling (current default)
  - Last token embedding
  - Special [COND] token (learned position)
  - CLS-style pooling with projection

- [ ] **Foundation-like behavior** — Evaluate whether language improves transfer to unseen PDE regimes (new equation types, new coefficient ranges).

- [ ] **1D → 2D transfer** — Test if a model trained on 1D PDEs can transfer to 2D (Gray-Scott is already 2D, so this may require architectural changes).

- [ ] **Direct token-based conditioning** — Cross-attention with LLM token sequences (deprioritized per plan).

### Evaluation Metrics Still Needed

- [ ] **Rollout stability metric** — Blow-up rate (fraction of test trajectories where nRMSE exceeds a threshold)
- [ ] **OOD parameter error** — nRMSE on PDE parameters outside the training distribution
- [ ] **Resolution transfer error** — nRMSE when testing at higher spatial resolution than training

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Train baseline FNOs (no conditioning)
python train.py --phase 1

# Phase 2: Train language-conditioned FNOs (warm-started from Phase 1)
python train.py --phase 2

# Generate comparison figures
python plot.py
```

To change the conditioning method, edit `conditioning_method` in `config.py`:
```python
conditioning_method: str = "spectral_gating"  # or "film"
```

To change the LLM, edit `llm_model_name` in `config.py`:
```python
llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

---

## Dependencies

- **JAX / jaxlib** — automatic differentiation, vmap parallelism
- **APEBench** (≥0.1.1) — PDE benchmarking framework, scenarios, metrics
- **Equinox** — JAX neural network modules (via APEBench)
- **pdequinox** — PDE neural operator architectures (ClassicFNO)
- **Transformers** (≥4.47.0) — HuggingFace model loading
- **Flax** — Flax model backend for LLM embedding
- **matplotlib / seaborn / pandas** — visualization and data handling
