#!/bin/bash
# Submit the 8 CoT experiment jobs to SLURM.
#
# Experiment matrix: {FiLM, Spectral Gating} × {frozen, joint} × {declarative_long, cot_generated}
# Each sbatch job trains all 6 PDEs in one invocation with the given configuration.
#
# IMPORTANT: populate the CoT cache before submitting, so the 4 parallel
# cot_generated jobs read from disk instead of each regenerating the same
# texts (and racing on cache writes):
#
#     python tools/populate_cot_cache.py
#
# Usage:
#     bash sbatch_scripts/submit_cot.sh

set -euo pipefail

cd /home/sgoudarzi/lang_conditioned_neural_ops
mkdir -p slurm_logs

# ── Safety check: warn if cache is empty but cot jobs are being submitted ──
if [ ! -d "cache/cot_generated" ] || [ -z "$(ls -A cache/cot_generated 2>/dev/null)" ]; then
    echo "WARNING: cache/cot_generated/ is empty."
    echo "         The 4 cot_generated jobs will each try to generate CoT texts"
    echo "         in parallel, which wastes GPU time and can race on writes."
    echo ""
    echo "         Recommended: run  python tools/populate_cot_cache.py  first."
    echo ""
    read -rp "Submit anyway? [y/N] " response
    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        echo "Aborted."
        exit 1
    fi
fi

echo "Submitting 8 CoT experiment jobs..."
echo ""

# Spectral gating variants
JOB4=$(sbatch --parsable sbatch_scripts/job4_sg_frozen_decl_long.sh)
echo "Job 4 submitted: $JOB4  — sg frozen declarative_long"

JOB5=$(sbatch --parsable sbatch_scripts/job5_sg_frozen_cot.sh)
echo "Job 5 submitted: $JOB5  — sg frozen cot_generated"

JOB6=$(sbatch --parsable sbatch_scripts/job6_sg_joint_decl_long.sh)
echo "Job 6 submitted: $JOB6  — sg joint declarative_long"

JOB7=$(sbatch --parsable sbatch_scripts/job7_sg_joint_cot.sh)
echo "Job 7 submitted: $JOB7  — sg joint cot_generated"

# FiLM variants
JOB8=$(sbatch --parsable sbatch_scripts/job8_film_frozen_decl_long.sh)
echo "Job 8 submitted: $JOB8  — film frozen declarative_long"

JOB9=$(sbatch --parsable sbatch_scripts/job9_film_frozen_cot.sh)
echo "Job 9 submitted: $JOB9  — film frozen cot_generated"

JOB10=$(sbatch --parsable sbatch_scripts/job10_film_joint_decl_long.sh)
echo "Job 10 submitted: $JOB10 — film joint declarative_long"

JOB11=$(sbatch --parsable sbatch_scripts/job11_film_joint_cot.sh)
echo "Job 11 submitted: $JOB11 — film joint cot_generated"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "After all complete, run:  python plot.py"
