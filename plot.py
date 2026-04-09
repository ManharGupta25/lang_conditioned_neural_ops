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

_METHOD_PALETTE = {
    "baseline":            "#4C72B0",
    "film":                "#DD8452",
    "spectral_gating":     "#55A868",
    "film_joint":          "#C44E52",
    "spectral_gating_joint": "#8172B2",
}
_METHOD_LABELS = {
    "baseline":            "Baseline FNO",
    "film":                "FiLM Conditioned",
    "spectral_gating":     "Spectral Gating",
    "film_joint":          "FiLM Joint Fine-Tuned",
    "spectral_gating_joint": "SG Joint Fine-Tuned",
}


def _condition_to_method(condition: str) -> str:
    """Extract conditioning method from a condition label.
    'film_distilbert' -> 'film'
    'spectral_gating_tinyllama' -> 'spectral_gating'
    'film_tinyllama_joint' -> 'film_joint'
    'spectral_gating_tinyllama_joint' -> 'spectral_gating_joint'
    'baseline' -> 'baseline'
    """
    is_joint = condition.endswith("_joint")
    cond = condition.removesuffix("_joint") if is_joint else condition
    joint_suffix = "_joint" if is_joint else ""
    for method in ("spectral_gating", "film", "baseline"):
        if cond == method or cond.startswith(method + "_"):
            return method + joint_suffix
    return condition


def _condition_label(condition: str) -> str:
    """Human-readable label for a condition string (includes LLM if present)."""
    method = _condition_to_method(condition)
    base = _METHOD_LABELS.get(method, condition)
    # Append LLM name if encoded (e.g. 'film_distilbert' -> 'FiLM Conditioned (distilbert)')
    parts = condition.split("_", maxsplit=len(method.split("_")))
    llm_suffix = "_".join(parts[len(method.split("_")):])
    if llm_suffix:
        return f"{base} ({llm_suffix})"
    return base


def _condition_color(condition: str) -> str:
    method = _condition_to_method(condition)
    return _METHOD_PALETTE.get(method, "#999999")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all(cfg: Config, results_dir: pathlib.Path = None) -> pd.DataFrame:
    """
    Load and concatenate all result CSVs from results_dir.
    Picks up baseline + any Phase 2 CSVs present in the directory.
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    # PDEs that share a prefix with another PDE (e.g. phy_adv vs phy_adv_diff)
    # need special care so their globs don't bleed into each other.
    all_pdes = set(cfg.pde_scenarios)

    dfs = []
    for pde in cfg.pde_scenarios:
        # Other PDEs whose name starts with this pde — their files would be
        # incorrectly matched by the glob f"{pde}_*.csv".
        longer_pdes = {p for p in all_pdes if p != pde and p.startswith(pde + "_")}

        for csv_path in sorted(results_dir.glob(f"{pde}_*.csv")):
            # Skip files that actually belong to a longer-named PDE
            if any(csv_path.stem.startswith(p + "_") for p in longer_pdes):
                continue
            label = csv_path.stem[len(pde) + 1:]   # strip "<pde>_"
            df = pd.read_csv(csv_path)
            df["condition"] = label
            df["pde"] = pde
            dfs.append(df)

        if not (results_dir / f"{pde}_baseline.csv").exists():
            print(f"  [warn] no baseline CSV for {pde} in {results_dir}")

    if not dfs:
        raise FileNotFoundError(
            f"No result CSVs found in {results_dir}. Run train.py --phase 1 (and 2) first."
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

def plot_nrmse_rollout(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path, suffix: str = ""):
    """
    One subplot per PDE.
    x: test timestep   y: mean nRMSE
    Two lines (baseline / conditioned) with 95% CI shading across seeds.
    """
    df = melt_nrmse(data)
    df["Condition"] = df["condition"].map(_condition_label)
    conditions = df["condition"].unique()
    palette = {_condition_label(c): _condition_color(c) for c in conditions}

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
    out = save_dir / f"nrmse_rollout{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 — Training loss ──────────────────────────────────────────────────

def plot_train_loss(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path, suffix: str = ""):
    """
    One subplot per PDE.
    x: training step   y: MSE loss (log scale)
    Two lines with 95% CI shading across seeds.
    """
    df = melt_loss(data)
    if df.empty:
        print("  [warn] No loss columns found — skipping train_loss.png")
        return

    df["Condition"] = df["condition"].map(_condition_label)
    conditions = df["condition"].unique()
    palette = {_condition_label(c): _condition_color(c) for c in conditions}

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
    out = save_dir / f"train_loss{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 — Final nRMSE bar chart ─────────────────────────────────────────

def plot_final_nrmse_bar(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path, suffix: str = ""):
    """
    Grouped bar chart: PDE × condition at the final test timestep.
    Error bars = std across seeds.
    """
    df = melt_nrmse(data)
    final_t = df["time_step"].max()
    df = df[df["time_step"] == final_t].copy()
    df["Condition"] = df["condition"].map(_condition_label)
    df["PDE"] = df["pde"].map(_pde_title)
    conditions = df["condition"].unique()
    palette = {_condition_label(c): _condition_color(c) for c in conditions}

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
    out = save_dir / f"final_nrmse_bar{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument(
        "--results-dir", default=None,
        help="Override results directory (default: cfg.results_dir from config.py)",
    )
    parser.add_argument(
        "--figures-dir", default=None,
        help="Override figures output directory (default: cfg.figures_dir from config.py)",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Suffix appended to output filenames, e.g. 'baseline' -> nrmse_rollout_baseline.png",
    )
    args = parser.parse_args()

    cfg = Config()
    results_dir = pathlib.Path(args.results_dir) if args.results_dir else pathlib.Path(cfg.results_dir)
    figures_dir = pathlib.Path(args.figures_dir) if args.figures_dir else pathlib.Path(cfg.figures_dir)
    suffix = f"_{args.suffix}" if args.suffix else ""
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results ...")
    data = load_all(cfg, results_dir=results_dir)
    conditions = sorted(data["condition"].unique()) if "condition" in data.columns else []
    print(f"  PDEs: {data['pde'].nunique()}  "
          f"Seeds: {data['seed'].nunique()}  "
          f"Conditions: {conditions}")

    print(f"\nGenerating figures -> {figures_dir}/")
    plot_nrmse_rollout(data, cfg, figures_dir, suffix)
    plot_train_loss(data, cfg, figures_dir, suffix)
    plot_final_nrmse_bar(data, cfg, figures_dir, suffix)
    print("\nDone.")


if __name__ == "__main__":
    main()
