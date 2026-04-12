#!/usr/bin/env python
"""
Run the 8 missing Phase 2 experiments.

Grouped by conditioning config so each subprocess trains only the
PDEs that are actually missing for that config.

Usage:
    python run_missing_experiments.py          # run all 8 sequentially
    python run_missing_experiments.py --dry-run # print commands only
"""

import argparse
import subprocess
import sys

# ── Missing experiments ──────────────────────────────────────────────────────
# (conditioning_method, freeze_trunk, [missing PDEs])

MISSING = [
    # Job 1: spectral_gating, frozen trunk — 1 missing PDE
    ("spectral_gating", "true", ["phy_burgers"]),

    # Job 2: film, joint fine-tuning — 3 missing PDEs
    ("film", "false", ["phy_adv_diff", "phy_ks", "phy_burgers"]),

    # Job 3: spectral_gating, joint fine-tuning — 4 missing PDEs
    ("spectral_gating", "false", ["phy_adv", "phy_adv_diff", "phy_ks", "phy_burgers"]),
]


def build_command(method, freeze, scenarios):
    return [
        sys.executable, "train.py",
        "--phase", "2",
        "--conditioning-method", method,
        "--freeze-trunk", freeze,
        "--scenarios", *scenarios,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()

    for i, (method, freeze, scenarios) in enumerate(MISSING, 1):
        freeze_label = "frozen" if freeze == "true" else "joint"
        cmd = build_command(method, freeze, scenarios)
        print(f"\n{'='*65}")
        print(f"Job {i}/3: {method} ({freeze_label}) — {scenarios}")
        print(f"{'='*65}")
        print(f"  $ {' '.join(cmd)}\n")

        if not args.dry_run:
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"\nERROR: Job {i} failed with return code {result.returncode}")
                sys.exit(result.returncode)

    print("\nAll missing experiments complete. Run:  python plot.py")


if __name__ == "__main__":
    main()
