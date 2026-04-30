"""
plot.py — Generate comparison figures from Phase 1 and Phase 2 results.

Reads CSVs from results/ and produces three figures in figures/:

    nrmse_rollout.png      — nRMSE vs test timestep (log y-axis)
                             2×3 subplots, one per PDE, 95% CI across seeds,
                             shared legend below the grid.
    train_loss.png         — training loss vs update step (log y-axis)
                             2×3 subplots with shared legend.
    final_nrmse_bar.png    — final-timestep nRMSE bar chart
                             2×3 subplots (one per PDE) so each gets its own
                             y-scale and fair-error bars aren't crushed by
                             the PDE with the largest error.

Run after both training phases are complete:
    python plot.py
"""

import argparse
import math
import pathlib
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import Config

_cfg = Config()
RESULTS_DIR = pathlib.Path(_cfg.results_dir)
FIGURES_DIR = pathlib.Path(_cfg.figures_dir)


# ── Condition parsing ────────────────────────────────────────────────────────
#
# Condition strings come from the `condition` column in each CSV and match the
# Phase-2 label convention from config.phase2_label(). Examples:
#
#     baseline
#     film_distilbert-base
#     film_tinyllama-1.1b-chat-
#     film_tinyllama-1.1b-chat-_joint
#     film_tinyllama-1.1b-chat-_cot_generated
#     film_tinyllama-1.1b-chat-_cot_generated_joint
#     film_tinyllama-1.1b-chat-_declarative_long_joint
#     spectral_gating_tinyllama-1.1b-chat-_cot_generated
#
# The parser peels the suffixes in a fixed order: _joint first, then prompt
# style (cot_generated / declarative_long), then the method prefix. What's
# left after method+underscore is the LLM short name.

_PROMPT_STYLE_SUFFIXES = ("cot_generated", "declarative_long")
_METHOD_PREFIXES = ("spectral_gating", "film")

_METHOD_DISPLAY = {"baseline": "Baseline", "film": "FiLM", "spectral_gating": "SG"}
_PROMPT_DISPLAY = {
    "declarative":      "",          # default — omit from label
    "declarative_long": "decl-long",
    "cot_generated":    "CoT",
}


def _shorten_llm(llm_raw: str) -> str:
    """'tinyllama-1.1b-chat-' → 'TinyLlama'; 'distilbert-base' → 'DistilBERT'."""
    if llm_raw is None:
        return ""
    s = llm_raw.lower()
    if "tinyllama" in s: return "TinyLlama"
    if "distilbert" in s: return "DistilBERT"
    if "gpt" in s: return "GPT"
    if "llama" in s: return "Llama"
    if "mistral" in s: return "Mistral"
    return llm_raw.split("-")[0].title()


def parse_condition(condition: str) -> Dict[str, object]:
    """Extract {method, llm, prompt, joint} from a condition string."""
    if condition == "baseline":
        return {"method": "baseline", "llm": None, "prompt": "declarative", "joint": False}

    s = condition
    joint = s.endswith("_joint")
    if joint:
        s = s[: -len("_joint")]

    prompt = "declarative"
    for style in _PROMPT_STYLE_SUFFIXES:
        if s.endswith("_" + style):
            prompt = style
            s = s[: -(len(style) + 1)]
            break

    method, llm = s, None
    for m in _METHOD_PREFIXES:
        if s == m:
            method = m
            break
        if s.startswith(m + "_"):
            method = m
            llm = s[len(m) + 1:]
            break

    return {"method": method, "llm": llm, "prompt": prompt, "joint": joint}


def condition_label(condition: str) -> str:
    """Short human-readable label for a condition. Unique per (method,llm,prompt,joint)."""
    info = parse_condition(condition)
    if info["method"] == "baseline":
        return "Baseline"

    method = _METHOD_DISPLAY.get(info["method"], info["method"])
    parts: List[str] = []
    if info["llm"]:
        parts.append(_shorten_llm(info["llm"]))
    if _PROMPT_DISPLAY.get(info["prompt"]):
        parts.append(_PROMPT_DISPLAY[info["prompt"]])
    if info["joint"]:
        parts.append("joint")
    return f"{method} ({', '.join(parts)})" if parts else method


def _sort_key(condition: str) -> Tuple:
    """Canonical ordering so the same condition gets the same color/order
    in every figure. Baseline first, then by method, LLM, prompt, trunk."""
    info = parse_condition(condition)
    method_rank = {"baseline": 0, "film": 1, "spectral_gating": 2}.get(info["method"], 9)
    llm_rank = 0 if info["llm"] is None else (1 if "distilbert" in (info["llm"] or "") else 2)
    prompt_rank = {"declarative": 0, "declarative_long": 1, "cot_generated": 2}.get(info["prompt"], 9)
    joint_rank = 1 if info["joint"] else 0
    return (method_rank, llm_rank, prompt_rank, joint_rank)


# ── Color assignment ─────────────────────────────────────────────────────────

def build_palette(conditions: List[str]) -> Dict[str, str]:
    """One distinct color per condition, with baseline always black.
    Groups related conditions by using hues from the same method family."""
    ordered = sorted(set(conditions), key=_sort_key)

    palette: Dict[str, str] = {}
    # Separate baseline so it always reads as "the reference."
    non_baseline = [c for c in ordered if parse_condition(c)["method"] != "baseline"]
    if "baseline" in ordered:
        palette["baseline"] = "#000000"

    # Assign colors from a perceptually uniform palette sized to however many
    # non-baseline conditions exist.
    n = max(1, len(non_baseline))
    colors = sns.color_palette("husl", n_colors=n)
    for c, col in zip(non_baseline, colors):
        palette[c] = col
    return palette


# ── Data loading ─────────────────────────────────────────────────────────────

def load_all(cfg: Config, results_dir: pathlib.Path = None, baseline_dir: pathlib.Path = None) -> pd.DataFrame:
    """
    Load and concatenate all result CSVs.
    - Baseline CSVs loaded from baseline_dir (falls back to cfg.phase1_results_dir, then results_dir)
    - Phase 2 CSVs loaded from results_dir
    """
    if results_dir is None:
        results_dir = RESULTS_DIR
    if baseline_dir is None:
        baseline_dir = pathlib.Path(cfg.phase1_results_dir) if cfg.phase1_results_dir else results_dir

    all_pdes = set(cfg.pde_scenarios)
    dfs = []
    for pde in cfg.pde_scenarios:
        longer_pdes = {p for p in all_pdes if p != pde and p.startswith(pde + "_")}

        # Load baseline from baseline_dir
        baseline_csv = baseline_dir / f"{pde}_baseline.csv"
        if baseline_csv.exists():
            df = pd.read_csv(baseline_csv)
            df["condition"] = "baseline"
            df["pde"] = pde
            dfs.append(df)
        else:
            print(f"  [warn] no baseline CSV for {pde} in {baseline_dir}")

        # Load Phase 2 CSVs from results_dir (skip baseline — already loaded)
        for csv_path in sorted(results_dir.glob(f"{pde}_*.csv")):
            if any(csv_path.stem.startswith(p + "_") for p in longer_pdes):
                continue
            label = csv_path.stem[len(pde) + 1:]
            if label == "baseline":
                continue   # already loaded from baseline_dir
            df = pd.read_csv(csv_path)
            df["condition"] = label
            df["pde"] = pde
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No result CSVs found in {results_dir}. Run train.py --phase 1 (and 2) first."
        )
    return pd.concat(dfs, ignore_index=True)


# ── Melting helpers ──────────────────────────────────────────────────────────

def _id_cols(data: pd.DataFrame) -> list:
    return [c for c in ("seed", "pde", "condition") if c in data.columns]


def melt_nrmse(data: pd.DataFrame) -> pd.DataFrame:
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


# ── Shared styling ───────────────────────────────────────────────────────────

def _apply_grid(ax):
    ax.grid(True, alpha=0.25, linestyle="--", which="both")
    ax.spines[["top", "right"]].set_visible(False)


def _pde_title(pde_key: str) -> str:
    """'phy_burgers_sc' → 'Burgers (single-channel)'; 'phy_ks' → 'Kuramoto-Sivashinsky'."""
    nice = {
        "phy_burgers_sc": "Burgers (single-channel)",
        "phy_burgers":    "Burgers",
        "phy_diff":       "Diffusion",
        "phy_adv":        "Advection",
        "phy_adv_diff":   "Advection-Diffusion",
        "phy_ks":         "Kuramoto-Sivashinsky",
        "phy_gs":         "Gray-Scott",
        "phy_ks_cons":    "KS (conservative)",
        "phy_kdv":        "KdV",
        "phy_fisher":     "Fisher-KPP",
    }
    return nice.get(pde_key, pde_key.removeprefix("phy_").replace("_", " ").title())


def _grid_shape(n: int) -> Tuple[int, int]:
    """Prefer 2×N/2 for n>=4 so subplots get real estate; single row for n<=3."""
    if n <= 3:
        return 1, n
    if n <= 6:
        return 2, math.ceil(n / 2)
    return 3, math.ceil(n / 3)


def _make_grid(n_pdes: int, subplot_size: Tuple[float, float] = (5.0, 3.8)):
    rows, cols = _grid_shape(n_pdes)
    w, h = subplot_size
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(w * cols, h * rows),
        squeeze=False,
    )
    flat = axes.flatten()
    # Hide any leftover axes when n_pdes isn't a perfect fit.
    for ax in flat[n_pdes:]:
        ax.set_visible(False)
    return fig, flat[:n_pdes], rows, cols


def _shared_legend(fig, handles, labels, rows: int, cols: int):
    """Place one shared legend below the subplot grid."""
    n_items = len(labels)
    ncol = min(n_items, max(3, cols * 2))
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )


def _ordered_handles(ax, all_conditions: List[str], palette: Dict[str, str]):
    """Return legend handles/labels in canonical order (baseline first, etc.)."""
    handles, labels = ax.get_legend_handles_labels()
    # Canonical order across figures
    ordered_conds = sorted(set(all_conditions), key=_sort_key)
    label_to_cond = {condition_label(c): c for c in set(all_conditions)}
    order = []
    for c in ordered_conds:
        lbl = condition_label(c)
        if lbl in labels:
            order.append(labels.index(lbl))
    return [handles[i] for i in order], [labels[i] for i in order]


# ── Figure 1 — nRMSE rollout (log y) ─────────────────────────────────────────

def plot_nrmse_rollout(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path, suffix: str = ""):
    df = melt_nrmse(data)
    all_conditions = list(df["condition"].unique())
    palette_by_cond = build_palette(all_conditions)
    palette_by_label = {condition_label(c): palette_by_cond[c] for c in all_conditions}
    df["Condition"] = df["condition"].map(condition_label)

    fig, axes, rows, cols = _make_grid(len(cfg.pde_scenarios))

    for ax, pde in zip(axes, cfg.pde_scenarios):
        sub = df[df["pde"] == pde]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.lineplot(
            data=sub,
            x="time_step",
            y="mean_nRMSE",
            hue="Condition",
            hue_order=[condition_label(c) for c in sorted(all_conditions, key=_sort_key)],
            palette=palette_by_label,
            errorbar=("ci", 95),
            linewidth=1.4,
            ax=ax,
            legend=False,
        )
        ax.set_yscale("log")
        ax.set_title(_pde_title(pde), fontsize=11, fontweight="bold")
        ax.set_xlabel("Test Timestep")
        ax.set_ylabel("Mean nRMSE (log)")
        _apply_grid(ax)

    # Build legend once from a dummy invisible plot so every label is present
    dummy_fig, dummy_ax = plt.subplots()
    for cond in sorted(set(all_conditions), key=_sort_key):
        dummy_ax.plot([], [], color=palette_by_cond[cond], linewidth=2, label=condition_label(cond))
    handles, labels = dummy_ax.get_legend_handles_labels()
    plt.close(dummy_fig)

    _shared_legend(fig, handles, labels, rows, cols)
    fig.suptitle("Rollout Error: Baseline vs Language-Conditioned FNO", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = save_dir / f"nrmse_rollout{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 — Training loss (log y) ─────────────────────────────────────────

def plot_train_loss(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path, suffix: str = ""):
    df = melt_loss(data)
    if df.empty:
        print("  [warn] No loss columns found — skipping train_loss.png")
        return

    all_conditions = list(df["condition"].unique())
    palette_by_cond = build_palette(all_conditions)
    palette_by_label = {condition_label(c): palette_by_cond[c] for c in all_conditions}
    df["Condition"] = df["condition"].map(condition_label)

    fig, axes, rows, cols = _make_grid(len(cfg.pde_scenarios))

    for ax, pde in zip(axes, cfg.pde_scenarios):
        sub = df[df["pde"] == pde]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.lineplot(
            data=sub,
            x="update_step",
            y="train_loss",
            hue="Condition",
            hue_order=[condition_label(c) for c in sorted(all_conditions, key=_sort_key)],
            palette=palette_by_label,
            errorbar=("ci", 95),
            linewidth=1.4,
            ax=ax,
            legend=False,
        )
        ax.set_yscale("log")
        ax.set_title(_pde_title(pde), fontsize=11, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("MSE Loss (log)")
        _apply_grid(ax)

    dummy_fig, dummy_ax = plt.subplots()
    for cond in sorted(set(all_conditions), key=_sort_key):
        dummy_ax.plot([], [], color=palette_by_cond[cond], linewidth=2, label=condition_label(cond))
    handles, labels = dummy_ax.get_legend_handles_labels()
    plt.close(dummy_fig)

    _shared_legend(fig, handles, labels, rows, cols)
    fig.suptitle("Training Loss: Baseline vs Language-Conditioned FNO", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    out = save_dir / f"train_loss{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 — Faceted final-nRMSE bars ──────────────────────────────────────

def plot_final_nrmse_bar(data: pd.DataFrame, cfg: Config, save_dir: pathlib.Path, suffix: str = ""):
    """One subplot per PDE so each gets its own y-scale — fixes the problem
    where a PDE with much larger error (e.g. Diffusion) crushed every other
    PDE's bars into invisibility."""
    df = melt_nrmse(data)
    final_t = df["time_step"].max()
    df = df[df["time_step"] == final_t].copy()
    df["Condition"] = df["condition"].map(condition_label)

    all_conditions = list(df["condition"].unique())
    palette_by_cond = build_palette(all_conditions)
    palette_by_label = {condition_label(c): palette_by_cond[c] for c in all_conditions}
    ordered_labels = [condition_label(c) for c in sorted(all_conditions, key=_sort_key)]

    fig, axes, rows, cols = _make_grid(len(cfg.pde_scenarios), subplot_size=(6.0, 4.2))

    for ax, pde in zip(axes, cfg.pde_scenarios):
        sub = df[df["pde"] == pde]
        if sub.empty:
            ax.set_visible(False)
            continue
        sns.barplot(
            data=sub,
            x="Condition",
            y="mean_nRMSE",
            hue="Condition",
            order=ordered_labels,
            hue_order=ordered_labels,
            palette=palette_by_label,
            errorbar="sd",
            capsize=0.15,
            ax=ax,
            legend=False,
        )
        ax.set_title(_pde_title(pde), fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Mean nRMSE")
        # Per-subplot scale so each PDE is readable on its own terms.
        # Rotate x labels so long condition names don't overlap.
        ax.tick_params(axis="x", rotation=40)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
            tick.set_fontsize(8)
        _apply_grid(ax)

    dummy_fig, dummy_ax = plt.subplots()
    for cond in sorted(set(all_conditions), key=_sort_key):
        dummy_ax.bar([0], [0], color=palette_by_cond[cond], label=condition_label(cond))
    handles, labels = dummy_ax.get_legend_handles_labels()
    plt.close(dummy_fig)

    _shared_legend(fig, handles, labels, rows, cols)
    fig.suptitle(
        f"Final Mean nRMSE at Timestep {final_t}  (lower is better; error bars = std across seeds)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    out = save_dir / f"final_nrmse_bar{suffix}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument(
        "--conditions", default=None,
        help="Comma-separated substrings to include (baseline always included). "
             "E.g. --conditions distilbert,tinyllama",
    )
    parser.add_argument(
        "--exclude", default=None,
        help="Comma-separated substrings to exclude after --conditions filter. "
             "E.g. --exclude cot_generated,declarative_long",
    )
    args = parser.parse_args()

    cfg = Config()
    results_dir = pathlib.Path(args.results_dir) if args.results_dir else pathlib.Path(cfg.results_dir)
    baseline_dir = pathlib.Path(cfg.phase1_results_dir) if cfg.phase1_results_dir else results_dir
    figures_dir = pathlib.Path(args.figures_dir) if args.figures_dir else pathlib.Path(cfg.figures_dir)
    suffix = f"_{args.suffix}" if args.suffix else ""
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading results ...")
    data = load_all(cfg, results_dir=results_dir, baseline_dir=baseline_dir)

    if args.conditions:
        filters = [s.strip() for s in args.conditions.split(",") if s.strip()]
        data = data[
            data["condition"].apply(
                lambda c: c == "baseline" or any(f in c for f in filters)
            )
        ]

    if args.exclude:
        excludes = [s.strip() for s in args.exclude.split(",") if s.strip()]
        data = data[
            data["condition"].apply(
                lambda c: c == "baseline" or not any(e in c for e in excludes)
            )
        ]

    conditions = sorted(data["condition"].unique()) if "condition" in data.columns else []
    print(f"  PDEs: {data['pde'].nunique()}  "
          f"Seeds: {data['seed'].nunique()}  "
          f"Conditions ({len(conditions)}):")
    for c in sorted(conditions, key=_sort_key):
        print(f"    - {c!r:60s} → {condition_label(c)}")

    print(f"\nGenerating figures -> {figures_dir}/")
    plot_nrmse_rollout(data, cfg, figures_dir, suffix)
    plot_train_loss(data, cfg, figures_dir, suffix)
    plot_final_nrmse_bar(data, cfg, figures_dir, suffix)
    print("\nDone.")


if __name__ == "__main__":
    main()
