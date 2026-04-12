#!/bin/bash
# Submit all 3 missing-experiment jobs to SLURM.
# All jobs are independent and can run in parallel on separate GPUs.
#
# Usage:
#   bash sbatch_scripts/submit_all.sh

set -euo pipefail

cd /home/sgoudarzi/lang_conditioned_neural_ops
mkdir -p slurm_logs

echo "Submitting 3 jobs (8 missing experiments total)..."
echo ""

JOB1=$(sbatch --parsable sbatch_scripts/job1_sg_frozen.sh)
echo "Job 1 submitted: $JOB1  — spectral_gating frozen (phy_burgers)"

JOB2=$(sbatch --parsable sbatch_scripts/job2_film_joint.sh)
echo "Job 2 submitted: $JOB2  — film joint (phy_adv_diff, phy_ks, phy_burgers)"

JOB3=$(sbatch --parsable sbatch_scripts/job3_sg_joint.sh)
echo "Job 3 submitted: $JOB3  — sg joint (phy_adv, phy_adv_diff, phy_ks, phy_burgers)"

echo ""
echo "Monitor with:  squeue -u \$USER"
echo "After all complete, run:  python plot.py"
