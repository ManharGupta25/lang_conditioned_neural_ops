"""
train.py — Language-conditioned FNO training via APEBench.

Two separate phases, run independently:

    python train.py --phase 1        Train baseline FNOs, save checkpoints + CSVs
    python train.py --phase 2        Load Phase 1 checkpoints, warm-start
                                     ConditionedFNOs, train with LLM frozen

After both phases:

    python plot.py                   Generate comparison figures

Phase 1 must complete before Phase 2.
Checkpoints : checkpoints/<run>/<scenario_key>_baseline.eqx          (Phase 1)
             checkpoints/<run>/<scenario_key>_<method>_<llm>.eqx    (Phase 2)
Result CSVs : results/<run>/<scenario_key>_baseline.csv             (Phase 1)
             results/<run>/<scenario_key>_<method>_<llm>.csv        (Phase 2)

Multi-seed:
    APEBench vmaps over seeds for Phase 1 (plain scenario()).
    Phase 2 builds one warm-started ConditionedFNO per seed (Python loop),
    stacks them, then vmaps the training loop — giving the same parallelism.

To change the LLM, edit Config.llm_model_name in config.py.
"""

import argparse
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from apebench.scenarios import scenario_dict

from config import Config
from data.embeddings import precompute_embeddings
from models.conditioned_fno import ConditionedFNO




# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scenario(scenario_key: str, cfg: Config, optim_config: str = None):
    return scenario_dict[scenario_key](
        num_spatial_dims=cfg.num_spatial_dims,
        num_points=cfg.num_points,
        num_train_samples=cfg.num_train_samples,
        num_test_samples=cfg.num_test_samples,
        train_temporal_horizon=cfg.train_temporal_horizon,
        test_temporal_horizon=cfg.test_temporal_horizon,
        batch_size=cfg.batch_size,
        optim_config=optim_config or cfg.optim_config(),
    )


def _rollout_profile(data: pd.DataFrame) -> list:
    """Full nRMSE-vs-timestep list, averaged across seeds."""
    cols = sorted(c for c in data.columns if c.startswith("mean_nRMSE_"))
    return [float(data[c].mean()) for c in cols]


def _final_metric(data: pd.DataFrame) -> float:
    """Mean nRMSE at the last test timestep, averaged across seeds."""
    cols = sorted(c for c in data.columns if c.startswith("mean_nRMSE_"))
    return float(data[cols[-1]].mean()) if cols else float("nan")


def _save_csv(data: pd.DataFrame, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    print(f"  CSV saved  -> {path}")


def _ckpt(scenario_key: str, label: str, cfg: Config) -> pathlib.Path:
    return pathlib.Path(cfg.checkpoints_dir) / f"{scenario_key}_{label}.eqx"


# ── Phase 2 internals — multi-seed warm-starting ─────────────────────────────

def _load_baseline_batch(scenario_key: str, cfg: Config) -> eqx.Module:
    """
    Load the Phase 1 multi-seed checkpoint.
    Re-creates the batched model structure then deserialises weights.
    Reads from cfg.phase1_checkpoints_dir if set, otherwise cfg.checkpoints_dir.
    """
    ckpt_dir = pathlib.Path(cfg.phase1_checkpoints_dir) if cfg.phase1_checkpoints_dir else pathlib.Path(cfg.checkpoints_dir)
    ckpt = ckpt_dir / f"{scenario_key}_baseline.eqx"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No Phase 1 checkpoint at {ckpt}.\n"
            "Run  python train.py --phase 1  first."
        )
    scenario = _make_scenario(scenario_key, cfg)
    # Build a [num_seeds]-batched template by vmapping the constructor
    template_batch = eqx.filter_vmap(
        lambda s: scenario.get_neural_stepper(
            task_config="predict",
            network_config=cfg.fno_network_config(),
            key=jax.random.PRNGKey(s),
        )
    )(jnp.arange(cfg.num_seeds))
    return eqx.tree_deserialise_leaves(str(ckpt), template_batch)


def _build_warmstarted_batch(
    phase1_batch: eqx.Module,
    z_embed,
    cfg: Config,
) -> ConditionedFNO:
    """
    Build a [num_seeds]-batched ConditionedFNO where the FNO trunk of each
    seed is warm-started from the corresponding Phase 1 checkpoint weights.
    The conditioning components (z_proj, film/spectral layers) are freshly initialised.

    Steps:
        1. Loop over seeds in Python to extract per-seed Phase 1 weights.
        2. Build a ConditionedFNO for each seed and copy the FNO trunk.
        3. Stack all seeds' leaf arrays along axis 0 to form a batched model.
    """
    z = jnp.array(z_embed)
    cond_fno_list = []

    for i in range(cfg.num_seeds):
        # Extract seed i's Phase 1 model (drop the batch axis)
        p1_i = jax.tree.map(lambda x: x[i] if eqx.is_array(x) else x, phase1_batch)

        # Build ConditionedFNO with random conditioning weights
        c_i = ConditionedFNO(
            num_spatial_dims=cfg.num_spatial_dims,
            num_channels=1,          # all current scenarios are 1-channel
            num_modes=cfg.fno_modes,
            hidden_channels=cfg.fno_hidden,
            num_blocks=cfg.fno_blocks,
            activation=jax.nn.gelu,
            z_embed=z,
            cond_dim=cfg.z_cond_dim,
            conditioning_method=cfg.conditioning_method,
            sg_mlp_width=cfg.sg_mlp_width,
            sg_mlp_depth=cfg.sg_mlp_depth,
            freeze_trunk=cfg.freeze_fno_trunk,
            key=jax.random.PRNGKey(i),
        )

        # Warm-start: overwrite the FNO trunk with Phase 1 weights
        c_i = eqx.tree_at(
            lambda m: (m.fno_lifting, m.fno_blocks, m.fno_projection),
            c_i,
            (p1_i.lifting, list(p1_i.blocks), p1_i.projection),
        )
        cond_fno_list.append(c_i)

    # Stack leaf arrays along a new leading seed axis
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0) if eqx.is_array(xs[0]) else xs[0], *cond_fno_list)


def _build_result_df(
    metric_trj_s: dict,
    loss_batch,
    scenario_key: str,
    net_config: str,
    cfg: Config,
    condition: str,
    record_loss_every: int = 100,
) -> pd.DataFrame:
    """
    Build an APEBench-compatible wide DataFrame from raw metric and loss arrays
    so that apebench.melt_metrics / melt_loss work on Phase 2 results.

    metric_trj_s["mean_nRMSE"] : (num_seeds, test_temporal_horizon)
    loss_batch                  : (num_seeds, num_loss_entries)
    """
    record_every = record_loss_every
    rows = []
    nrmse_arr = metric_trj_s["mean_nRMSE"]   # (num_seeds, T)
    for seed_i in range(cfg.num_seeds):
        row = {
            "seed"           : seed_i,
            "scenario"       : scenario_key,
            "task"           : "predict",
            "net"            : net_config,
            "train"          : "one",
            "scenario_kwargs": "{}",
            "condition"      : condition,
            "pde"            : scenario_key,
        }
        # nRMSE rollout columns:  mean_nRMSE_0001 ... mean_nRMSE_XXXX
        for t in range(cfg.test_temporal_horizon):
            row[f"mean_nRMSE_{t + 1:04d}"] = float(nrmse_arr[seed_i, t])
        # Loss columns:  train_loss_000000, train_loss_000100, ...
        for step_i in range(loss_batch.shape[1]):
            actual_step = step_i * record_every
            row[f"train_loss_{actual_step:06d}"] = float(loss_batch[seed_i, step_i])
        rows.append(row)
    return pd.DataFrame(rows)


# ── Phase 1 — Baseline FNOs ───────────────────────────────────────────────────

def run_phase1(cfg: Config):
    """
    Train one baseline FNO per PDE (no language conditioning).
    APEBench vmaps over cfg.num_seeds seeds automatically.
    Saves model checkpoints and APEBench-format CSVs.
    """
    results_dir = pathlib.Path(cfg.results_dir)
    pathlib.Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"PHASE 1  —  Baseline FNO  ({cfg.num_seeds} seeds per PDE)")
    print("=" * 65)

    for scenario_key in cfg.pde_scenarios:
        print(f"\n  Training: {scenario_key}")
        scenario = _make_scenario(scenario_key, cfg)

        data, model = scenario(
            task_config="predict",
            network_config=cfg.fno_network_config(),
            train_config="one",
            start_seed=0,
            num_seeds=cfg.num_seeds,
        )

        # Tag for plot.py
        data["condition"] = "baseline"
        data["pde"]       = scenario_key

        eqx.tree_serialise_leaves(str(_ckpt(scenario_key, "baseline", cfg)), model)
        print(f"  Checkpoint -> {_ckpt(scenario_key, 'baseline', cfg)}")
        _save_csv(data, results_dir / f"{scenario_key}_baseline.csv")

        # Stability readout
        profile      = _rollout_profile(data)
        sampled      = profile[::max(1, len(profile) // 10)]
        nrmse_final  = _final_metric(data)
        print(f"  nRMSE rollout (sampled): {[f'{v:.3f}' for v in sampled]}")
        print(f"  Final nRMSE: {nrmse_final:.4f}  "
              f"({'OK' if nrmse_final < 0.5 else 'CHECK — high error'})")

    print("\nPhase 1 complete. Run  python train.py --phase 2  when ready.")


# ── Phase 2 — Conditioned FNOs with warm-start ────────────────────────────────

def run_phase2(cfg: Config):
    """
    For each PDE:
      1. Load Phase 1 multi-seed checkpoint  (shape: [num_seeds, ...])
      2. Precompute z from frozen LLM.
      3. Build one ConditionedFNO per seed, warm-starting the FNO trunk
         from the corresponding Phase 1 seed's weights.
      4. Stack into a batched model, then eqx.filter_vmap the training loop
         — same GPU parallelism as APEBench's native multi-seed.
      5. Evaluate, save checkpoint and APEBench-format CSV.
    """
    results_dir = pathlib.Path(cfg.results_dir)
    pathlib.Path(cfg.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"PHASE 2  —  Conditioned FNO  ({cfg.num_seeds} seeds per PDE)")
    print(f"LLM       : {cfg.llm_model_name}")
    print(f"Trunk     : {'frozen' if cfg.freeze_fno_trunk else 'jointly fine-tuned'}")
    print(f"Steps     : {cfg.phase2_train_steps}")
    print("=" * 65)

    print("\nPrecomputing language embeddings ...")
    embeddings, embed_dim = precompute_embeddings(
        cfg.pde_scenarios, cfg.llm_model_name,
    )
    cfg.z_embed_dim = embed_dim
    print(f"z_embed_dim = {embed_dim}\n")

    summary = {}
    for scenario_key in cfg.pde_scenarios:
        print(f"  Training: {scenario_key}")
        scenario = _make_scenario(scenario_key, cfg, optim_config=cfg.phase2_optim_config())

        # ── Step 1: load Phase 1 weights ──────────────────────────────────────
        phase1_batch = _load_baseline_batch(scenario_key, cfg)
        print(f"  Phase 1 checkpoint loaded ({cfg.num_seeds} seeds).")

        # ── Step 2: build warm-started batched ConditionedFNO ─────────────────
        z = embeddings[scenario_key]
        cond_fno_batch = _build_warmstarted_batch(phase1_batch, z, cfg)
        print(f"  FNO trunk warm-started; conditioning={cfg.conditioning_method}.")

        # ── Step 3: vmap training over seeds ──────────────────────────────────
        trainer      = scenario.get_trainer(train_config="one")
        shuffle_keys = jax.random.split(jax.random.PRNGKey(99), cfg.num_seeds)

        def train_one(cond_fno, shuffle_key):
            trained, loss_history, _ = trainer(
                cond_fno,
                shuffle_key,
                return_loss_history=True,
                record_loss_every=scenario.record_loss_every,
                spawn_tqdm=False,
            )
            return trained, loss_history

        trunk_status = "frozen" if cfg.freeze_fno_trunk else "jointly fine-tuned"
        print(f"  Running vmapped training (FNO trunk {trunk_status}) ...")
        trained_batch, loss_batch = eqx.filter_vmap(train_one)(
            cond_fno_batch, shuffle_keys
        )

        # ── Step 4: evaluate metrics across seeds ─────────────────────────────
        metric_trj_s = scenario.perform_tests(trained_batch)

        # ── Step 5: build DataFrame, save ─────────────────────────────────────
        label = cfg.phase2_label()   # e.g. "film_distilbert" or "spectral_gating_tinyllama"
        data = _build_result_df(
            metric_trj_s, loss_batch,
            scenario_key, cfg.cfno_network_config(), cfg,
            condition=label,
            record_loss_every=scenario.record_loss_every,
        )
        eqx.tree_serialise_leaves(str(_ckpt(scenario_key, label, cfg)), trained_batch)
        print(f"  Checkpoint -> {_ckpt(scenario_key, label, cfg)}")
        _save_csv(data, results_dir / f"{scenario_key}_{label}.csv")

        nrmse_final = _final_metric(data)
        print(f"  Final nRMSE: {nrmse_final:.4f}\n")
        summary[scenario_key] = nrmse_final

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 65)
    print("PHASE 2 SUMMARY  vs Phase 1 baseline")
    print("=" * 65)
    header = f"{'PDE':<22}  {'Baseline':>10}  {'Conditioned':>12}  {'Delta':>8}"
    print(header)
    print("-" * len(header))
    for key in cfg.pde_scenarios:
        cond = summary.get(key, float("nan"))
        p1_csv = results_dir / f"{key}_baseline.csv"
        base = _final_metric(pd.read_csv(p1_csv)) if p1_csv.exists() else float("nan")
        delta = cond - base
        sign = "+" if delta >= 0 else ""
        print(f"{key:<22}  {base:>10.4f}  {cond:>12.4f}  {sign}{delta:>7.4f}")
    print(f"Conditioning method: {cfg.conditioning_method}")

    print(f"\nRun  python plot.py  to generate comparison figures.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Language-conditioned FNO (APEBench + JAX)"
    )
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], required=True,
        help="1 = train baseline FNOs;  2 = warm-start + train conditioned FNOs",
    )
    args = parser.parse_args()

    cfg = Config()
    print("=" * 65)
    print("Language-Conditioned Neural Operators  (APEBench + JAX)")
    print("=" * 65)
    print(f"PDEs        : {cfg.pde_scenarios}")
    print(f"Seeds       : {cfg.num_seeds}")
    print(f"FNO         : modes={cfg.fno_modes}  hidden={cfg.fno_hidden}  "
          f"blocks={cfg.fno_blocks}")
    print(f"Train steps : {cfg.num_train_steps}  |  batch={cfg.batch_size}")

    if args.phase == 1:
        run_phase1(cfg)
    else:
        run_phase2(cfg)


if __name__ == "__main__":
    main()
