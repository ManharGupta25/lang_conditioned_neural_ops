from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    # ── PDE scenarios ─────────────────────────────────────────────────────────
    # APEBench scenario keys to train on
    pde_scenarios: List[str] = field(default_factory=lambda: [
        "phy_burgers_sc",   # 1D Burgers (nonlinear advection + diffusion)
        "phy_diff",         # 1D Diffusion (purely dissipative)
        "phy_adv",          # 1D Advection (pure transport)
        "phy_adv_diff",     # 1d advection diffusion
        "phy_ks",           # 1D Kuramoto-Shivashinsky
        "phy_burgers",      # 1D Burgers (multi-dimensional)
        #"phy_gs"
    ])

    # ── Spatial setup ─────────────────────────────────────────────────────────
    num_spatial_dims: int = 1
    num_points: int = 128

    # ── Language model ────────────────────────────────────────────────────────
    # Any HuggingFace model name. FlaxAutoModel is used when the model has a
    # Flax implementation (LLaMA, Mistral, DistilBERT etc.); PyTorch fallback
    # otherwise. z_embed_dim is auto-detected from the loaded model's hidden_dim.
    #
    # Small / fast (development):
    #   "distilbert-base-uncased"         768-dim  encoder  66M params
    #   "gpt2"                            768-dim  decoder  124M params
    #   "EleutherAI/gpt-neo-125m"         768-dim  decoder  125M params
    #
    # Medium (better language prior):
    #   "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  2048-dim decoder  1.1B params
    #   "facebook/opt-350m"               512-dim  decoder  350M params
    #
    # Large (strongest prior — needs GPU + HF token for gated models):
    #   "meta-llama/Llama-2-7b-hf"        4096-dim decoder  7B params
    #   "mistralai/Mistral-7B-v0.1"       4096-dim decoder  7B params
    #
    llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    # z_embed_dim is set automatically from the loaded model; do not set manually.
    # It is written here as a placeholder that gets filled in train.py after loading.
    z_embed_dim: Optional[int] = None

    # ── FNO architecture ──────────────────────────────────────────────────────
    # Shared between baseline FNO and conditioned FNO.
    fno_modes: int = 12
    fno_hidden: int = 64
    fno_blocks: int = 6
    fno_activation: str = "gelu"

    # ── Conditioning ──────────────────────────────────────────────────────────
    # z_cond_dim: projection from z_embed_dim -> this dim, then fed into the
    # conditioning layers. Keep small relative to fno_hidden to avoid over-conditioning.
    z_cond_dim: int = 64

    # Primary ablation axis: which method to use for injecting z into the FNO.
    #   "film"            — FiLM: (1 + gamma) * x + beta  per channel
    #   "spectral_gating" — per-frequency gate in Fourier space
    conditioning_method: str = "spectral_gating"

    # Spectral gating MLP hyperparameters (only used when conditioning_method="spectral_gating")
    sg_mlp_width: int = 64    # hidden layer width (default: same as z_cond_dim)
    sg_mlp_depth: int = 2     # number of hidden layers

    # ── Training (Phase 1) ────────────────────────────────────────────────────
    num_train_steps: int = 20000
    num_train_samples: int = 1000
    batch_size: int = 50
    train_temporal_horizon: int = 50

    # ── Training (Phase 2) ────────────────────────────────────────────────────
    # Steps when FNO trunk is frozen — only conditioning layers train.
    phase2_train_steps: int = 20000

    # Steps for joint fine-tuning (freeze_fno_trunk=False). Fewer than
    # phase2_train_steps to reduce risk of trunk drifting from Phase 1 solution.
    phase2_joint_train_steps: int = 2000

    # If True, FNO trunk (lifting, blocks, projection) is frozen — only z_proj
    # and conditioning layers are trained. Clean ablation: measures what language
    # conditioning contributes independently of trunk adaptation.
    # If False, full joint fine-tuning — trunk + conditioning trained together.
    # Use a lower learning rate and fewer steps to avoid trunk drift.
    freeze_fno_trunk: bool = True

    # ── Evaluation ────────────────────────────────────────────────────────────
    num_test_samples: int = 100
    test_temporal_horizon: int = 200

    # ── Seeds ─────────────────────────────────────────────────────────────────
    # APEBench vmaps over seeds automatically — gives mean ± CI for free.
    num_seeds: int = 3

    # ── Output ────────────────────────────────────────────────────────────────
    results_dir: str = "results/fno_12_64_6_gelu_train_20000_1000_50_50_test_100"       # CSVs saved here for plotting
    figures_dir: str = "figures/fno_12_64_6_gelu_train_20000_1000_50_50_test_100"       # figures saved here by plot.py
    checkpoints_dir: str = "checkpoints/fno_12_64_6_gelu_train_20000_1000_50_50_test_100"  # model checkpoints

    # If set, Phase 2 loads Phase 1 baseline checkpoints from this directory
    # instead of checkpoints_dir. Useful when pointing Phase 2 at a fixed
    # Phase 1 run stored in a different folder.
    # Leave as None to use checkpoints_dir for both reading and writing.
    phase1_checkpoints_dir: Optional[str] = "checkpoints/fno_12_64_6_gelu_train_20000_1000_50_50_test_100"

    # ── Derived APEBench config strings ──────────────────────────────────────

    def llm_short_name(self) -> str:
        """Compact LLM identifier safe for use in filenames.
        e.g. 'distilbert-base-uncased' -> 'distilbert'
             'TinyLlama/TinyLlama-1.1B-Chat-v1.0' -> 'tinyllama'
             'meta-llama/Llama-2-7b-hf' -> 'llama-2-7b'
        """
        name = self.llm_model_name.split("/")[-1]   # strip org prefix
        name = name.lower().split("-uncased")[0].split("-cased")[0]
        return name.split("_")[0][:20]              # cap length

    def phase2_label(self) -> str:
        """Label used for Phase 2 checkpoint and CSV filenames.
        Encodes conditioning method, LLM, and freeze state.
        e.g. 'film_distilbert', 'spectral_gating_tinyllama',
             'film_tinyllama_joint' (when freeze_fno_trunk=False)
        """
        suffix = "_joint" if not self.freeze_fno_trunk else ""
        return f"{self.conditioning_method}_{self.llm_short_name()}{suffix}"

    def fno_network_config(self) -> str:
        """Network config string for the plain baseline FNO."""
        return f"fno;{self.fno_modes};{self.fno_hidden};{self.fno_blocks};{self.fno_activation}"

    def cfno_network_config(self) -> str:
        """Network config string for the conditioned FNO (registered as 'cfno')."""
        return f"cfno;{self.fno_modes};{self.fno_hidden};{self.fno_blocks};{self.fno_activation}"

    def optim_config(self) -> str:
        """Optimizer config string for Phase 1."""
        warmup = max(200, self.num_train_steps // 6)
        return f"adam;{self.num_train_steps};warmup_cosine;0.0;1e-3;{warmup}"

    def phase2_optim_config(self) -> str:
        """Optimizer config for Phase 2.
        Frozen trunk: standard LR (1e-3), phase2_train_steps.
        Joint fine-tuning: lower LR (5e-4), phase2_joint_train_steps — avoids trunk drift.
        """
        steps = self.phase2_train_steps if self.freeze_fno_trunk else self.phase2_joint_train_steps
        warmup = max(100, steps // 6)
        lr = "1e-3" if self.freeze_fno_trunk else "5e-4"
        return f"adam;{steps};warmup_cosine;0.0;{lr};{warmup}"
