"""
plot.py — Generate comparison figures from Phase 1 and Phase 2 results.

Reads CSVs from results/ and produces three figures in figures/:

    nrmse_rollout.png      — nRMSE vs test timestep, baseline vs conditioned
                             (one subplot per PDE, shaded 95% CI across seeds)
    train_loss.png         — training loss vs update step
                             (one subplot per PDE, shaded 95% CI across seeds)
    final_nrmse_bar.png    — grouped bar chart of final-timestep nRMSE
                             (error bars = std across seeds)

Run after both training phases are complete:
    python plot.py
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import Config

_cfg = Config()
RESULTS_DIR = pathlib.Path(_cfg.results_dir)
FIGURES_DIR = pathlib.Path(_cfg.figures_dir)

PALETTE = {
    "baseline":         "#4C72B0",
    "film":             "#DD8452",
    "spectral_gating":  "#55A868",
}
LABELS = {
    "baseline":         "Baseline FNO",
    "film":             "FiLM Conditioned",
    "spectral_gating":  "Spectral Gating",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(cfg: Config, phase: int = 0) -> pd.DataFrame:
    """
    Load and concatenate result CSVs.

    phase=1 : baseline only
    phase=0 : baseline + all available Phase 2 conditioning method CSVs
    """
    phase2_methods = [m for m in LABELS if m != "baseline"]
    conditions = ["baseline"] if phase == 1 else ["baseline"] + phase2_methods

    dfs = []
    for pde in cfg.pde_scenarios:
        for condition in conditions:
            path = RESULTS_DIR / f"{pde}_{condition}.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["condition"] = condition
                df["pde"] = pde
                dfs.append(df)
            else:
                if condition == "baseline":
                    print(f"  [warn] missing {path} — skipping")
    if not dfs:
        raise FileNotFoundError(
            "No result CSVs found in results/. Run train.py --phase 1 (and 2) first."
        )
    return pd.concat(dfs, ignore_index=True)


# ── Melting helpers ───────────────────────────────────────────────────────────

def _id_cols(data: pd.DataFrame) -> list:
    """Return whichever identifying columns are present."""
    candidates = ["seed", "pde", "condition"]
    return [c for c in candidates if c in data.columns]


def melt_nrmse(data: pd.DataFrame) -> pd.DataFrame:
    """Wide → long for mean_nRMSE_XXXX columns."""
    stub = "mean_nRMSE_"
    value_cols = sorted(c for c in data.columns if c.startswith(stub))
    melted = data[_id_cols(data) + value_cols].melt(
        id_vars=_id_cols(data),
        value_vars=value_cols,
        var_name="_col",
        value_name="mean_nRMSE",
    )
    melted["time_step"] = melted["_col"].str.removeprefix(stub).astype(int)
    return melted.drop(columns="_col")


def melt_loss(data: pd.DataFrame) -> pd.DataFrame:
    """Wide → long for train_loss_XXXXXX columns."""
    stub = "train_loss_"
    value_cols = sorted(c for c in data.columns if c.startswith(stub))
    if not value_cols:
        return pd.DataFrame()
    melted = data[_id_cols(data) + value_cols].melt(
        id_vars=_id_cols(data),
        value_vars=value_cols,
        var_name="_col",
        value_name="train_loss",
    )
    melted["update_step"] = melted["_col"].str.removeprefix(stub).astype(int)
    return melted.drop(columns="_col")


# ── Shared plot styling ───────────────────────────────────────────────────────

def _apply_grid(ax):
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)


def _pde_title(pde_key: str) -> str:
    """'phy_burgers_sc' → 'Burgers Sc'"""
    return pde_key.removeprefix("phy_").replace("_", " ").title()


def _make_fig(n_pdes: int, height: float = 4.0):
    fig, axes = plt.subplots(
        1, n_pdes, figsize=(5 * n_pdes, height), squeeze=False
    )
    return fig, axes[0]   # axes[0] is the 1-D array of axes


# ── Figure 1 — nRMSE rollout ──────────────────────────────────────────────────

def plot_nrmse_rollout(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path):
    """
    One subplot per PDE.
    x: test timestep   y: mean nRMSE
    Two lines (baseline / conditioned) with 95% CI shading across seeds.
    """
    df = melt_nrmse(data)
    df["Condition"] = df["condition"].map(lambda c: LABELS.get(c, c))
    palette = {LABELS.get(k, k): v for k, v in PALETTE.items()}

    fig, axes = _make_fig(len(cfg.pde_scenarios))

    for ax, pde in zip(axes, cfg.pde_scenarios):
        sns.lineplot(
            data=df[df["pde"] == pde],
            x="time_step",
            y="mean_nRMSE",
            hue="Condition",
            palette=palette,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(_pde_title(pde), fontsize=12, fontweight="bold")
        ax.set_xlabel("Test Timestep")
        ax.set_ylabel("Mean nRMSE")
        ax.legend(title="", fontsize=9)
        _apply_grid(ax)

    fig.suptitle(
        "Rollout Error: Baseline vs Language-Conditioned FNO",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = save_dir / "nrmse_rollout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 — Training loss ──────────────────────────────────────────────────

def plot_train_loss(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path):
    """
    One subplot per PDE.
    x: training step   y: MSE loss (log scale)
    Two lines with 95% CI shading across seeds.
    """
    df = melt_loss(data)
    if df.empty:
        print("  [warn] No loss columns found — skipping train_loss.png")
        return

    df["Condition"] = df["condition"].map(lambda c: LABELS.get(c, c))
    palette = {LABELS.get(k, k): v for k, v in PALETTE.items()}

    fig, axes = _make_fig(len(cfg.pde_scenarios))

    for ax, pde in zip(axes, cfg.pde_scenarios):
        sns.lineplot(
            data=df[df["pde"] == pde],
            x="update_step",
            y="train_loss",
            hue="Condition",
            palette=palette,
            errorbar=("ci", 95),
            ax=ax,
        )
        ax.set_title(_pde_title(pde), fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss")
        ax.set_yscale("log")
        ax.legend(title="", fontsize=9)
        _apply_grid(ax)

    fig.suptitle(
        "Training Loss: Baseline vs Language-Conditioned FNO",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out = save_dir / "train_loss.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 — Final nRMSE bar chart ─────────────────────────────────────────

def plot_final_nrmse_bar(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path):
    """
    Grouped bar chart: PDE × condition at the final test timestep.
    Error bars = std across seeds.
    """
    df = melt_nrmse(data)
    final_t = df["time_step"].max()
    df = df[df["time_step"] == final_t].copy()
    df["Condition"] = df["condition"].map(lambda c: LABELS.get(c, c))
    df["PDE"] = df["pde"].map(_pde_title)
    palette = {LABELS.get(k, k): v for k, v in PALETTE.items()}

    fig, ax = plt.subplots(figsize=(max(5, len(cfg.pde_scenarios) * 2.5), 4))
    sns.barplot(
        data=df,
        x="PDE",
        y="mean_nRMSE",
        hue="Condition",
        palette=palette,
        errorbar="sd",
        capsize=0.08,
        ax=ax,
    )
    ax.set_title(
        f"Final Mean nRMSE at Timestep {final_t}  (lower is better)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("Mean nRMSE")
    ax.legend(title="", fontsize=9)
    _apply_grid(ax)

    fig.tight_layout()
    out = save_dir / "final_nrmse_bar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--phase", type=int, choices=[1], default=0,
        help="1 = plot Phase 1 baseline only; omit to include all available results",
    )
    args = parser.parse_args()

    cfg = Config()
    FIGURES_DIR.mkdir(exist_ok=True)

    print("Loading results ...")
    data = load_all(cfg, phase=args.phase)
    conditions = sorted(data["condition"].unique()) if "condition" in data.columns else []
    print(f"  PDEs: {data['pde'].nunique()}  "
          f"Seeds: {data['seed'].nunique()}  "
          f"Conditions: {conditions}")

    print("\nGenerating figures -> figures/")
    plot_nrmse_rollout(data, cfg, FIGURES_DIR)
    plot_train_loss(data, cfg, FIGURES_DIR)
    plot_final_nrmse_bar(data, cfg, FIGURES_DIR)
    print("\nDone.")


if __name__ == "__main__":
    main()
